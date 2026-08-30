#include "scheme_ii_fused.h"
#include "scheme_ii_int192.h"
#include "scheme_ii_moduli.h"
#include "scheme_ii_mma_group.cuh"
#include "int8_gemm.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace ozaki {

// SchemeII fused executor. The cuBLAS backend is staged; the custom backend
// can group moduli, fuse a partial CRT epilogue (M8.3 speed path), or run the
// all-moduli CRT epilogue (M8.4 control). --use_true_fused_kernel with
// group_size=4 selects M8.3, not full input-to-output fusion. A/B residue
// quantization stays staged unless --scheme_ii_prologue_fusion tile_local.

static int fused_mod_inverse(int a, int m) {
    int m0 = m, y = 0, x = 1;
    if (m == 1) return 0;
    while (a > 1) {
        int q = a / m;
        int t = m;
        m = a % m;
        a = t;
        t = y;
        y = x - q * y;
        x = t;
    }
    if (x < 0) x += m0;
    return x;
}

static uint32_t fused_mod_inverse_u32(uint32_t a, uint32_t m) {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)m;
    int64_t newr = (int64_t)(a % m);
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = newt;
        newt = t - q * newt;
        t = tmp;
        tmp = newr;
        newr = r - q * newr;
        r = tmp;
    }
    if (t < 0) t += (int64_t)m;
    return (uint32_t)t;
}

__device__ __forceinline__ double fused_pow2i_fast(int e) {
    int biased = e + 1023;
    if (biased > 0 && biased < 2047) {
        return __longlong_as_double((unsigned long long)biased << 52);
    }
    return ldexp(1.0, e);
}

static __device__ __forceinline__ int8_t fused_residue_i8(int64_t v_int, int p) {
    int rem = (int)(v_int % p);
    if (rem < 0) rem += p;
    if (rem > p / 2) rem -= p;
    return (int8_t)rem;
}

static __global__ void fused_compute_exp_A_kernel(const double* __restrict__ A, int* __restrict__ eA, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const double* row = A + (size_t)i * k;
    double max_val = 0.0;
    int kk = 0;
    for (; kk + 3 < k; kk += 4) {
        max_val = fmax(max_val, fabs(row[kk]));
        max_val = fmax(max_val, fabs(row[kk + 1]));
        max_val = fmax(max_val, fabs(row[kk + 2]));
        max_val = fmax(max_val, fabs(row[kk + 3]));
    }
    for (; kk < k; ++kk) max_val = fmax(max_val, fabs(row[kk]));
    int exp = 0;
    if (max_val > 0.0) frexp(max_val, &exp);
    eA[i] = exp;
}

static __global__ void fused_compute_exp_B_kernel(const double* __restrict__ B, int* __restrict__ eB, int k, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double max_val = 0.0;
    for (int kk = 0; kk < k; ++kk) {
        max_val = fmax(max_val, fabs(B[(size_t)kk * n + j]));
    }
    int exp = 0;
    if (max_val > 0.0) frexp(max_val, &exp);
    eB[j] = exp;
}

static __global__ void fused_quantize_A_kernel(const double* __restrict__ A, int8_t* __restrict__ A_mod,
                                              const int* __restrict__ eA,
                                              int m, int k, int m_pad, int k_pad, int num_moduli,
                                              const int32_t* __restrict__ primes, int S) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= m_pad || kk >= k_pad) return;

    int64_t v_int[4] = {0, 0, 0, 0};
    if (i < m) {
        double scale = fused_pow2i_fast(S - eA[i]);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int col = kk + t;
            if (col < k) v_int[t] = llround(A[(size_t)i * k + col] * scale);
        }
    }

    int p_reg[20];
    for (int p_idx = 0; p_idx < num_moduli; ++p_idx) p_reg[p_idx] = primes[p_idx];

    for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
        int p = p_reg[p_idx];
        char4 out;
        out.x = fused_residue_i8(v_int[0], p);
        out.y = fused_residue_i8(v_int[1], p);
        out.z = fused_residue_i8(v_int[2], p);
        out.w = fused_residue_i8(v_int[3], p);
        *reinterpret_cast<char4*>(&A_mod[(size_t)p_idx * m_pad * k_pad + i * k_pad + kk]) = out;
    }
}

static __global__ void fused_quantize_B_kernel(const double* __restrict__ B, int8_t* __restrict__ B_mod,
                                              const int* __restrict__ eB,
                                              int k, int n, int k_pad, int n_pad, int num_moduli,
                                              const int32_t* __restrict__ primes, int S) {
    int j = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    if (kk >= k_pad || j >= n_pad) return;

    int64_t v_int[4] = {0, 0, 0, 0};
    if (kk < k) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int col = j + t;
            if (col < n) {
                v_int[t] = llround(B[(size_t)kk * n + col] * fused_pow2i_fast(S - eB[col]));
            }
        }
    }

    int p_reg[20];
    for (int p_idx = 0; p_idx < num_moduli; ++p_idx) p_reg[p_idx] = primes[p_idx];

    for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
        int p = p_reg[p_idx];
        char4 out;
        out.x = fused_residue_i8(v_int[0], p);
        out.y = fused_residue_i8(v_int[1], p);
        out.z = fused_residue_i8(v_int[2], p);
        out.w = fused_residue_i8(v_int[3], p);
        *reinterpret_cast<char4*>(&B_mod[(size_t)p_idx * k_pad * n_pad + kk * n_pad + j]) = out;
    }
}

static __global__ void fused_crt_reconstruct_kernel(const int32_t* __restrict__ C_mod, double* __restrict__ C,
                                                   const int* __restrict__ eA, const int* __restrict__ eB,
                                                   int m, int n, int m_pad, int n_pad, int num_moduli, int S,
                                                   const int32_t* __restrict__ primes, const int32_t* __restrict__ inv_table,
                                                   Int192 M, Int192 M_half, const Int192* __restrict__ P) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        int v[20];
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            int r = C_mod[(size_t)p_idx * m_pad * n_pad + i * n_pad + j];
            int p = primes[p_idx];
            r = r % p;
            if (r < 0) r += p;
            v[p_idx] = r;
        }

        for (int p_idx = 0; p_idx < num_moduli - 1; ++p_idx) {
            int v_ii = v[p_idx];
            for (int q_idx = p_idx + 1; q_idx < num_moduli; ++q_idx) {
                int p = primes[q_idx];
                long long diff = v[q_idx] - v_ii;
                diff = diff % p;
                if (diff < 0) diff += p;
                v[q_idx] = (int)((diff * inv_table[p_idx * 20 + q_idx]) % p);
            }
        }

        Int192 X;
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            X.add_mul_scalar(P[p_idx], (uint32_t)v[p_idx]);
        }

        bool is_neg = X.is_greater_than(M_half);
        if (is_neg) {
            Int192 diff;
            uint64_t borrow = 0;
            #pragma unroll
            for (int k_idx = 0; k_idx < 6; ++k_idx) {
                long long d = (long long)M.limbs[k_idx] - (long long)borrow - (long long)X.limbs[k_idx];
                if (d < 0) {
                    d += 4294967296LL;
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                diff.limbs[k_idx] = (uint32_t)d;
            }
            X = diff;
        }

        double C_int_dbl = X.to_double();
        if (is_neg) C_int_dbl = -C_int_dbl;

        int exp_sum = 2 * S - eA[i] - eB[j];
        C[(size_t)i * n + j] = C_int_dbl * fused_pow2i_fast(-exp_sum);
    }
}

static void destroy_scheme_ii_graph(AdaptiveOzakiEngine* engine) {
    if (!engine) return;
    if (engine->d_scheme_ii_graph_exec) {
        cudaGraphExecDestroy(engine->d_scheme_ii_graph_exec);
        engine->d_scheme_ii_graph_exec = nullptr;
    }
    if (engine->d_scheme_ii_graph) {
        cudaGraphDestroy(engine->d_scheme_ii_graph);
        engine->d_scheme_ii_graph = nullptr;
    }
    engine->d_scheme_ii_graph_valid = false;
    engine->d_scheme_ii_graph_custom_mma = false;
}

// Host scalars referenced by captured cuBLAS nodes must outlive the graph.
static int kFusedAlpha = 1;
static int kFusedBeta = 0;

static bool launch_custom_mma_groups(
    cudaStream_t stream,
    int m_pad, int n_pad, int k_pad, int num_moduli, int gsz,
    int8_t* d_A_mod, int8_t* d_B_mod, int32_t* d_C_mod,
    int32_t* d_primes, int32_t* d_inv_table,
    bool partial_crt, int* launches_out, int mma_variant)
{
    long long stride_B = (long long)k_pad * n_pad;
    long long stride_A = (long long)m_pad * k_pad;
    long long stride_C = (long long)m_pad * n_pad;
    int launches = 0;
    int p = 0;
    while (p < num_moduli) {
        int rem = num_moduli - p;
        int this_g = rem > gsz ? gsz : rem;
        if (this_g == 1 && !partial_crt) {
            cudaError_t st = int8_gemm_s8s8s32(
                d_A_mod + (size_t)p * stride_A,
                d_B_mod + (size_t)p * stride_B,
                d_C_mod + (size_t)p * stride_C,
                m_pad, n_pad, k_pad, stream);
            if (st != cudaSuccess) {
                printf("int8_gemm_s8s8s32 failed with %s at p_idx %d\n", cudaGetErrorString(st), p);
                return false;
            }
            p += 1;
            launches += 1;
            continue;
        }
        int tmpl = (this_g <= 2) ? 2 : 4;
        int32_t* C_out = partial_crt ? (d_C_mod + (size_t)launches * stride_C)
                                    : (d_C_mod + (size_t)p * stride_C);
        cudaError_t st = scheme_ii_launch_grouped_mma(
            d_A_mod + (size_t)p * stride_A,
            d_B_mod + (size_t)p * stride_B,
            C_out,
            m_pad, n_pad, k_pad,
            this_g, tmpl,
            d_primes, d_inv_table, p,
            partial_crt, stream, mma_variant);
        if (st != cudaSuccess) {
            printf("grouped MMA failed with %s at p_idx %d n_group %d\n",
                   cudaGetErrorString(st), p, this_g);
            return false;
        }
        p += this_g;
        launches += 1;
    }
    if (launches_out) *launches_out = launches;
    return true;
}

static bool launch_prologue_mma_groups(
    cudaStream_t stream,
    const double* d_A, const double* d_B, int32_t* d_C_mod,
    const int* d_eA, const int* d_eB,
    int m, int n, int k, int m_pad, int n_pad, int k_pad,
    int num_moduli, int gsz, int S,
    int32_t* d_primes, int mma_variant, int* launches_out)
{
    long long stride_C = (long long)m_pad * n_pad;
    int launches = 0;
    int p = 0;
    while (p < num_moduli) {
        int rem = num_moduli - p;
        int this_g = rem > gsz ? gsz : rem;
        if (this_g < 1) this_g = 1;
        uint32_t* C_out = reinterpret_cast<uint32_t*>(d_C_mod) + (size_t)launches * stride_C;
        cudaError_t st = scheme_ii_launch_grouped_mma_prologue(
            d_A, d_B, C_out, d_eA, d_eB,
            m, n, k, m_pad, n_pad, k_pad,
            this_g, S, d_primes, p, stream, mma_variant);
        if (st != cudaSuccess) {
            printf("tile-local prologue MMA failed with %s at p_idx %d n_group %d\n",
                   cudaGetErrorString(st), p, this_g);
            return false;
        }
        p += this_g;
        launches += 1;
    }
    if (launches_out) *launches_out = launches;
    return true;
}

static bool launch_fused_pipeline(cublasHandle_t handle, cudaStream_t stream,
                                  const double* d_A, const double* d_B, double* d_C,
                                  int m, int n, int k, int m_pad, int n_pad, int k_pad,
                                  int num_moduli, int S, Int192 M, Int192 M_half,
                                  int32_t* d_primes, int32_t* d_inv_table, Int192* d_P,
                                  int* d_eA, int* d_eB, int8_t* d_A_mod, int8_t* d_B_mod, int32_t* d_C_mod,
                                  cudaEvent_t* ev, bool profile, bool use_custom_mma,
                                  int group_size, bool use_partial_crt, bool use_true_fused,
                                  const int32_t* h_primes, int* gemm_launches_out,
                                  int mma_variant, bool use_prologue_fusion,
                                  cudaStream_t stream2, cudaEvent_t ev_b) {
    dim3 block2d(16, 16);
    cudaStream_t streamB = (stream2 && stream2 != stream) ? stream2 : stream;
    if (profile) cudaEventRecord(ev[0], stream);

    fused_compute_exp_A_kernel<<<(m + 255) / 256, 256, 0, stream>>>(d_A, d_eA, m, k);
    if (profile) cudaEventRecord(ev[1], stream);

    fused_compute_exp_B_kernel<<<(n + 255) / 256, 256, 0, streamB>>>(d_B, d_eB, k, n);
    if (profile) cudaEventRecord(ev[2], streamB);

    const bool skip_global_quantize = use_custom_mma && use_partial_crt && use_prologue_fusion && !use_true_fused;
    if (!skip_global_quantize) {
        dim3 qblock(32, 8);
        dim3 gridA(((k_pad / 4) + 31) / 32, (m_pad + 7) / 8);
        fused_quantize_A_kernel<<<gridA, qblock, 0, stream>>>(d_A, d_A_mod, d_eA, m, k, m_pad, k_pad, num_moduli, d_primes, S);
        if (profile) cudaEventRecord(ev[3], stream);

        dim3 gridB(((n_pad / 4) + 31) / 32, (k_pad + 7) / 8);
        fused_quantize_B_kernel<<<gridB, qblock, 0, streamB>>>(d_B, d_B_mod, d_eB, k, n, k_pad, n_pad, num_moduli, d_primes, S);
        if (profile) cudaEventRecord(ev[4], streamB);
    } else {
        if (profile) cudaEventRecord(ev[3], stream);
        if (profile) cudaEventRecord(ev[4], streamB);
    }

    if (streamB != stream && ev_b) {
        cudaEventRecord(ev_b, streamB);
        cudaStreamWaitEvent(stream, ev_b, 0);
    } else if (streamB != stream) {
        cudaStreamSynchronize(streamB);
    }

    long long stride_B = (long long)k_pad * n_pad;
    long long stride_A = (long long)m_pad * k_pad;
    long long stride_C = (long long)m_pad * n_pad;
    int launches = 0;

    if (use_true_fused) {
        cudaError_t st = scheme_ii_launch_true_fused(
            d_A_mod, d_B_mod, d_C, d_eA, d_eB,
            m, n, k, m_pad, n_pad, k_pad,
            num_moduli, S, d_primes, d_inv_table, M, M_half, d_P, stream);
        if (st != cudaSuccess) {
            printf("true fused MMA CRT failed with %s\n", cudaGetErrorString(st));
            return false;
        }
        launches = 1;
        if (gemm_launches_out) *gemm_launches_out = launches;
        if (profile) cudaEventRecord(ev[5], stream);
        if (profile) cudaEventRecord(ev[6], stream);
        return true;
    }

    if (use_custom_mma) {
        int gsz = group_size < 1 ? 1 : group_size;
        if (gsz > 4) gsz = 4;
        bool ok = skip_global_quantize
            ? launch_prologue_mma_groups(stream, d_A, d_B, d_C_mod, d_eA, d_eB,
                                         m, n, k, m_pad, n_pad, k_pad, num_moduli, gsz, S,
                                         d_primes, mma_variant, &launches)
            : launch_custom_mma_groups(stream, m_pad, n_pad, k_pad, num_moduli, gsz,
                                       d_A_mod, d_B_mod, d_C_mod, d_primes, d_inv_table,
                                       use_partial_crt, &launches, mma_variant);
        if (!ok) return false;
    } else {
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            cublasStatus_t st = cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n_pad, m_pad, k_pad,
                &kFusedAlpha,
                d_B_mod + (size_t)p_idx * stride_B, CUDA_R_8I, n_pad,
                d_A_mod + (size_t)p_idx * stride_A, CUDA_R_8I, k_pad,
                &kFusedBeta,
                d_C_mod + (size_t)p_idx * stride_C, CUDA_R_32I, n_pad,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                printf("cublasGemmEx failed with status %d at p_idx %d\n", (int)st, p_idx);
                return false;
            }
        }
        launches = num_moduli;
    }
    if (gemm_launches_out) *gemm_launches_out = launches;
    if (profile) cudaEventRecord(ev[5], stream);

    if (use_partial_crt) {
        int gsz = group_size < 1 ? 1 : group_size;
        if (gsz > 4) gsz = 4;
        uint32_t h_Mg[20] = {};
        int32_t h_inv_g[400] = {};
        Int192 h_Pg[20];
        int p = 0;
        for (int g = 0; g < launches; ++g) {
            int rem = num_moduli - p;
            int this_g = rem > gsz ? gsz : rem;
            if (this_g < 1) this_g = 1;
            uint32_t Mg = 1;
            for (int i = 0; i < this_g; ++i) Mg *= (uint32_t)h_primes[p + i];
            h_Mg[g] = Mg;
            if (g == 0) {
                h_Pg[0].add(1);
            } else {
                h_Pg[g] = h_Pg[g - 1];
                h_Pg[g].mul(h_Mg[g - 1]);
            }
            p += this_g;
        }
        for (int g = 0; g < launches; ++g) {
            for (int h = g + 1; h < launches; ++h) {
                h_inv_g[g * 20 + h] = (int32_t)fused_mod_inverse_u32(h_Mg[g], h_Mg[h]);
            }
        }
        cudaMemcpy(d_primes, h_Mg, launches * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inv_table, h_inv_g, 400 * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_P, h_Pg, launches * sizeof(Int192), cudaMemcpyHostToDevice);
        scheme_ii_launch_combine_partial_crt(
            reinterpret_cast<uint32_t*>(d_C_mod), d_C, d_eA, d_eB, m, n, m_pad, n_pad,
            launches, S, reinterpret_cast<uint32_t*>(d_primes), d_inv_table,
            M, M_half, d_P, stream);
    } else {
        dim3 gridC((n + 15) / 16, (m + 15) / 16);
        fused_crt_reconstruct_kernel<<<gridC, block2d, 0, stream>>>(
            d_C_mod, d_C, d_eA, d_eB, m, n, m_pad, n_pad, num_moduli, S, d_primes, d_inv_table, M, M_half, d_P);
    }
    if (profile) cudaEventRecord(ev[6], stream);
    return true;
}

void execute_scheme_ii_fused(cublasHandle_t handle, const double* d_A, const double* d_B, double* d_C,
                             int m, int n, int k, const OzakiConfig& config, AdaptiveOzakiEngine* engine) {
    if (!engine) return;

    PrecisionContract& contract = engine->mutablePrecisionContract();
    contract.used_true_fused_kernel = false;
    contract.used_cuda_graph = false;
    contract.used_scheme_ii_fused = false;
    contract.used_custom_mma = false;
    const bool use_custom_mma = (config.scheme_ii_gemm_backend == SchemeIIGemmBackend::CustomMMA);
    contract.scheme_ii_backend = use_custom_mma ? "custom_mma" : "cublas";
    int gemm_launches = 0;

    int num_moduli = engine->getSchemePlan().num_moduli;
    if (num_moduli <= 0) num_moduli = config.scheme_ii_num_moduli;
    if (num_moduli > 20) num_moduli = 20;
    if (num_moduli < 2) {
        contract.failure_reason = "scheme_ii_num_moduli must be between 2 and 20";
        return;
    }

    if (!engine->d_scheme_ii_A_mod || !engine->d_scheme_ii_B_mod || !engine->d_scheme_ii_C_mod ||
        !engine->d_scheme_ii_eA || !engine->d_scheme_ii_eB || !engine->d_scheme_ii_primes ||
        !engine->d_scheme_ii_inv_table || !engine->d_scheme_ii_P) {
        contract.failure_reason = "SchemeII_Fused workspace is not allocated";
        return;
    }

    int group_size = config.scheme_ii_moduli_group_size;
    bool use_true_fused = false;
    bool use_partial_crt = false;
    if (use_custom_mma && config.scheme_ii_use_true_fused_kernel) {
        if (group_size <= 0 || group_size >= num_moduli) {
            // M8.4 all-moduli true fused control path.
            use_true_fused = true;
            group_size = num_moduli;
        } else {
            // M8.3/M9 experimental speed path: grouped MMA + partial CRT.
            use_partial_crt = true;
            if (group_size < 2) group_size = 2;
            if (group_size > 4) group_size = 4;
        }
    } else if (use_custom_mma) {
        if (group_size < 1) group_size = 1;
        if (group_size > 4) group_size = 4;
    } else {
        group_size = 1;
    }
    const int mma_variant = config.scheme_ii_mma_variant;
    const bool use_prologue_fusion = use_partial_crt && (config.scheme_ii_prologue_fusion == 1);

    int m_pad = ((m + 127) / 128) * 128;
    int n_pad = ((n + 127) / 128) * 128;
    int k_pad = use_custom_mma ? ((k + 63) / 64) * 64 : ((k + 31) / 32) * 32;

    int32_t h_primes[20];
    int32_t h_inv_table[400];
    for (int i = 0; i < 400; ++i) h_inv_table[i] = 0;
    for (int i = 0; i < num_moduli; ++i) {
        h_primes[i] = MODULI_INT8[i];
        for (int j = i + 1; j < num_moduli; ++j) {
            h_inv_table[i * 20 + j] = fused_mod_inverse(MODULI_INT8[i], MODULI_INT8[j]);
        }
    }

    Int192 M;
    M.add(1);
    for (int i = 0; i < num_moduli; ++i) M.mul((uint32_t)MODULI_INT8[i]);

    Int192 M_half = M;
    M_half.shr1();

    Int192 h_P[20];
    h_P[0].add(1);
    for (int i = 1; i < num_moduli; ++i) {
        h_P[i] = h_P[i - 1];
        h_P[i].mul((uint32_t)MODULI_INT8[i - 1]);
    }

    double max_val = M.to_double() / (2.0 * std::max(k, 1));
    int S = (int)std::floor(std::log2(max_val) / 2.0);
    if (S > 53) S = 53;
    if (S < 0) S = 0;

    int32_t* d_primes = engine->d_scheme_ii_primes;
    int32_t* d_inv_table = engine->d_scheme_ii_inv_table;
    Int192* d_P = engine->d_scheme_ii_P;
    int* d_eA = engine->d_scheme_ii_eA;
    int* d_eB = engine->d_scheme_ii_eB;
    int8_t* d_A_mod = engine->d_scheme_ii_A_mod;
    int8_t* d_B_mod = engine->d_scheme_ii_B_mod;
    int32_t* d_C_mod = engine->d_scheme_ii_C_mod;

    cudaMemcpy(d_primes, h_primes, num_moduli * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_table, h_inv_table, 400 * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, h_P, num_moduli * sizeof(Int192), cudaMemcpyHostToDevice);

    if (!engine->d_scheme_ii_stream) {
        cudaStreamCreate(&engine->d_scheme_ii_stream);
    }
    if (!engine->d_scheme_ii_stream2) {
        cudaStreamCreate(&engine->d_scheme_ii_stream2);
    }
    if (!engine->d_scheme_ii_ev_b) {
        cudaEventCreateWithFlags(&engine->d_scheme_ii_ev_b, cudaEventDisableTiming);
    }
    cudaStream_t stream = engine->d_scheme_ii_stream;
    cudaStream_t stream2 = engine->d_scheme_ii_stream2;
    cublasSetStream(handle, stream);

    const bool profile = config.enable_stage_profiling;
    cudaEvent_t ev[7] = {};
    if (profile) {
        for (int i = 0; i < 7; ++i) cudaEventCreate(&ev[i]);
    }

    bool used_graph = false;
    bool pipeline_ok = true;
    // cudaFuncSetAttribute inside custom MMA is not graph-capturable.
    const bool want_graph = config.scheme_ii_use_cuda_graph &&
                            !config.scheme_ii_use_true_fused_kernel &&
                            !use_custom_mma;
    if (want_graph) {
        bool need_capture =
            !engine->d_scheme_ii_graph_valid ||
            engine->d_scheme_ii_graph_m != m ||
            engine->d_scheme_ii_graph_n != n ||
            engine->d_scheme_ii_graph_k != k ||
            engine->d_scheme_ii_graph_num_moduli != num_moduli ||
            engine->d_scheme_ii_graph_A != d_A ||
            engine->d_scheme_ii_graph_B != d_B ||
            engine->d_scheme_ii_graph_C != d_C ||
            engine->d_scheme_ii_graph_custom_mma != use_custom_mma;

        if (need_capture) {
            destroy_scheme_ii_graph(engine);
            cudaStreamSynchronize(stream);
            cudaError_t cap_st = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (cap_st == cudaSuccess) {
                launch_fused_pipeline(handle, stream, d_A, d_B, d_C, m, n, k, m_pad, n_pad, k_pad,
                                      num_moduli, S, M, M_half, d_primes, d_inv_table, d_P,
                                      d_eA, d_eB, d_A_mod, d_B_mod, d_C_mod, nullptr, false,
                                      use_custom_mma, group_size, use_partial_crt, use_true_fused,
                                      h_primes, &gemm_launches, mma_variant, use_prologue_fusion,
                                      nullptr, nullptr);
                cudaGraph_t graph = nullptr;
                cap_st = cudaStreamEndCapture(stream, &graph);
                if (cap_st == cudaSuccess && graph) {
                    cudaGraphExec_t exec = nullptr;
                    cudaError_t inst_st = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                    if (inst_st == cudaSuccess && exec) {
                        engine->d_scheme_ii_graph = graph;
                        engine->d_scheme_ii_graph_exec = exec;
                        engine->d_scheme_ii_graph_valid = true;
                        engine->d_scheme_ii_graph_m = m;
                        engine->d_scheme_ii_graph_n = n;
                        engine->d_scheme_ii_graph_k = k;
                        engine->d_scheme_ii_graph_num_moduli = num_moduli;
                        engine->d_scheme_ii_graph_A = d_A;
                        engine->d_scheme_ii_graph_B = d_B;
                        engine->d_scheme_ii_graph_C = d_C;
                        engine->d_scheme_ii_graph_custom_mma = use_custom_mma;
                    } else {
                        if (graph) cudaGraphDestroy(graph);
                        destroy_scheme_ii_graph(engine);
                    }
                } else {
                    destroy_scheme_ii_graph(engine);
                }
            }
        }

        if (engine->d_scheme_ii_graph_valid && engine->d_scheme_ii_graph_exec) {
            if (profile) cudaEventRecord(ev[0], stream);
            cudaError_t launch_st = cudaGraphLaunch(engine->d_scheme_ii_graph_exec, stream);
            if (launch_st == cudaSuccess) {
                used_graph = true;
                if (profile) cudaEventRecord(ev[6], stream);
            } else {
                destroy_scheme_ii_graph(engine);
            }
        }
    }

    if (!used_graph) {
        pipeline_ok = launch_fused_pipeline(handle, stream, d_A, d_B, d_C, m, n, k, m_pad, n_pad, k_pad,
                                            num_moduli, S, M, M_half, d_primes, d_inv_table, d_P,
                                            d_eA, d_eB, d_A_mod, d_B_mod, d_C_mod, ev, profile,
                                            use_custom_mma, group_size, use_partial_crt, use_true_fused,
                                            h_primes, &gemm_launches, mma_variant, use_prologue_fusion,
                                            stream2, engine->d_scheme_ii_ev_b);
    }

    if (profile) {
        if (pipeline_ok) cudaEventSynchronize(ev[6]);
        contract.stages.clear();
        if (!pipeline_ok) {
            // GEMM failed before the CRT event was recorded.
        } else if (used_graph) {
            float total = 0;
            cudaEventElapsedTime(&total, ev[0], ev[6]);
            contract.stages.push_back({"scale_exp_A", 0.0, 0});
            contract.stages.push_back({"scale_exp_B", 0.0, 0});
            contract.stages.push_back({"quantize_A", 0.0, 0});
            contract.stages.push_back({"quantize_B", 0.0, 0});
            contract.stages.push_back({"int8_gemm", total, 0});
            contract.stages.push_back({"crt_undo", 0.0, 0});
        } else {
            float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;
            cudaEventElapsedTime(&t1, ev[0], ev[1]);
            cudaEventElapsedTime(&t2, ev[1], ev[2]);
            cudaEventElapsedTime(&t3, ev[2], ev[3]);
            cudaEventElapsedTime(&t4, ev[3], ev[4]);
            cudaEventElapsedTime(&t5, ev[4], ev[5]);
            cudaEventElapsedTime(&t6, ev[5], ev[6]);
            contract.stages.push_back({"scale_exp_A", t1, 0});
            contract.stages.push_back({"scale_exp_B", t2, 0});
            contract.stages.push_back({"quantize_A", t3, 0});
            contract.stages.push_back({"quantize_B", t4, 0});
            contract.stages.push_back({"int8_gemm", t5, 0});
            contract.stages.push_back({"crt_undo", t6, 0});
        }
        for (int i = 0; i < 7; ++i) {
            if (ev[i]) cudaEventDestroy(ev[i]);
        }
    }

    cudaStreamSynchronize(stream);
    cublasSetStream(handle, 0);

    if (!pipeline_ok) {
        contract.failure_reason = use_custom_mma
            ? "custom MMA int8_gemm_s8s8s32 failed"
            : "cublasGemmEx failed in SchemeII_Fused";
        contract.used_scheme_ii_fused = false;
        contract.used_custom_mma = false;
        return;
    }

    contract.used_scheme_ii_fused = true;
    contract.used_true_fused_kernel = use_true_fused;
    contract.used_partial_crt_epilogue = use_partial_crt;
    contract.used_custom_mma = use_custom_mma;
    contract.used_cuda_graph = used_graph;
    contract.scheme_ii_backend = use_custom_mma ? "custom_mma" : "cublas";
    contract.scheme_ii_group_size = use_true_fused ? num_moduli : group_size;
    contract.scheme_ii_gemm_launches = gemm_launches;
    contract.scheme_ii_mma_variant = use_partial_crt
        ? (mma_variant <= 0 ? scheme_ii_default_mma_variant() : mma_variant)
        : 0;
    contract.scheme_ii_prologue_fusion = use_prologue_fusion ? "tile_local" : "none";
    contract.num_moduli = num_moduli;
    if (contract.fallback_mode.empty()) contract.fallback_mode = "None";
}

} // namespace ozaki
