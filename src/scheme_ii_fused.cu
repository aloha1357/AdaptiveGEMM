#include "scheme_ii_fused.h"
#include "scheme_ii_int192.h"
#include "scheme_ii_moduli.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace ozaki {

// Staged / semi-fused SchemeII path. Kernels are local copies so this
// translation unit does not depend on relocatable device code against
// scheme_ii_gpu.cu. This is not a true register-level fused CRT kernel.

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

static __global__ void fused_compute_exp_A_kernel(const double* A, int* eA, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double max_val = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            double v = fabs(A[(size_t)i * k + kk]);
            if (v > max_val) max_val = v;
        }
        int exp = 0;
        if (max_val > 0.0) frexp(max_val, &exp);
        eA[i] = exp;
    }
}

static __global__ void fused_compute_exp_B_kernel(const double* B, int* eB, int k, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double max_val = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            double v = fabs(B[(size_t)kk * n + j]);
            if (v > max_val) max_val = v;
        }
        int exp = 0;
        if (max_val > 0.0) frexp(max_val, &exp);
        eB[j] = exp;
    }
}

static __global__ void fused_quantize_A_kernel(const double* __restrict__ A, int8_t* __restrict__ A_mod,
                                              const int* __restrict__ eA,
                                              int m, int k, int m_pad, int k_pad, int num_moduli,
                                              const int32_t* __restrict__ primes, int S) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && kk < k) {
        double val = A[(size_t)i * k + kk];
        int exp = eA[i];
        double scaled = val * ldexp(1.0, S - exp);
        int64_t v_int = llround(scaled);
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            int p = primes[p_idx];
            int rem = (int)(v_int % p);
            if (rem < 0) rem += p;
            if (rem > p / 2) rem -= p;
            A_mod[(size_t)p_idx * m_pad * k_pad + i * k_pad + kk] = (int8_t)rem;
        }
    } else if (i < m_pad && kk < k_pad) {
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            A_mod[(size_t)p_idx * m_pad * k_pad + i * k_pad + kk] = 0;
        }
    }
}

static __global__ void fused_quantize_B_kernel(const double* __restrict__ B, int8_t* __restrict__ B_mod,
                                              const int* __restrict__ eB,
                                              int k, int n, int k_pad, int n_pad, int num_moduli,
                                              const int32_t* __restrict__ primes, int S) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    if (kk < k && j < n) {
        double val = B[(size_t)kk * n + j];
        int exp = eB[j];
        double scaled = val * ldexp(1.0, S - exp);
        int64_t v_int = llround(scaled);
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            int p = primes[p_idx];
            int rem = (int)(v_int % p);
            if (rem < 0) rem += p;
            if (rem > p / 2) rem -= p;
            B_mod[(size_t)p_idx * k_pad * n_pad + kk * n_pad + j] = (int8_t)rem;
        }
    } else if (kk < k_pad && j < n_pad) {
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            B_mod[(size_t)p_idx * k_pad * n_pad + kk * n_pad + j] = 0;
        }
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
        C[(size_t)i * n + j] = C_int_dbl * ldexp(1.0, -exp_sum);
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
}

// Host scalars referenced by captured cuBLAS nodes must outlive the graph.
static int kFusedAlpha = 1;
static int kFusedBeta = 0;

static void launch_fused_pipeline(cublasHandle_t handle, cudaStream_t stream,
                                  const double* d_A, const double* d_B, double* d_C,
                                  int m, int n, int k, int m_pad, int n_pad, int k_pad,
                                  int num_moduli, int S, Int192 M, Int192 M_half,
                                  int32_t* d_primes, int32_t* d_inv_table, Int192* d_P,
                                  int* d_eA, int* d_eB, int8_t* d_A_mod, int8_t* d_B_mod, int32_t* d_C_mod,
                                  cudaEvent_t* ev, bool profile) {
    const int block_size = 256;
    if (profile) cudaEventRecord(ev[0], stream);
    fused_compute_exp_A_kernel<<<(m + block_size - 1) / block_size, block_size, 0, stream>>>(d_A, d_eA, m, k);
    if (profile) cudaEventRecord(ev[1], stream);

    fused_compute_exp_B_kernel<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(d_B, d_eB, k, n);
    if (profile) cudaEventRecord(ev[2], stream);

    dim3 block2d(16, 16);
    dim3 gridA((k_pad + 15) / 16, (m_pad + 15) / 16);
    fused_quantize_A_kernel<<<gridA, block2d, 0, stream>>>(d_A, d_A_mod, d_eA, m, k, m_pad, k_pad, num_moduli, d_primes, S);
    if (profile) cudaEventRecord(ev[3], stream);

    dim3 gridB((n_pad + 15) / 16, (k_pad + 15) / 16);
    fused_quantize_B_kernel<<<gridB, block2d, 0, stream>>>(d_B, d_B_mod, d_eB, k, n, k_pad, n_pad, num_moduli, d_primes, S);
    if (profile) cudaEventRecord(ev[4], stream);

    long long stride_B = (long long)k_pad * n_pad;
    long long stride_A = (long long)m_pad * k_pad;
    long long stride_C = (long long)m_pad * n_pad;

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
            return;
        }
    }
    if (profile) cudaEventRecord(ev[5], stream);

    dim3 gridC((n + 15) / 16, (m + 15) / 16);
    fused_crt_reconstruct_kernel<<<gridC, block2d, 0, stream>>>(
        d_C_mod, d_C, d_eA, d_eB, m, n, m_pad, n_pad, num_moduli, S, d_primes, d_inv_table, M, M_half, d_P);
    if (profile) cudaEventRecord(ev[6], stream);
}

void execute_scheme_ii_fused(cublasHandle_t handle, const double* d_A, const double* d_B, double* d_C,
                             int m, int n, int k, const OzakiConfig& config, AdaptiveOzakiEngine* engine) {
    if (!engine) return;

    PrecisionContract& contract = engine->mutablePrecisionContract();
    contract.used_true_fused_kernel = false;
    contract.used_cuda_graph = false;
    contract.used_scheme_ii_fused = false;

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

    int m_pad = ((m + 127) / 128) * 128;
    int n_pad = ((n + 127) / 128) * 128;
    int k_pad = ((k + 31) / 32) * 32;

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
    cudaStream_t stream = engine->d_scheme_ii_stream;
    cublasSetStream(handle, stream);

    const bool profile = config.enable_stage_profiling;
    cudaEvent_t ev[7] = {};
    if (profile) {
        for (int i = 0; i < 7; ++i) cudaEventCreate(&ev[i]);
    }

    bool used_graph = false;
    const bool want_graph = config.scheme_ii_use_cuda_graph && !config.scheme_ii_use_true_fused_kernel;
    if (want_graph) {
        bool need_capture =
            !engine->d_scheme_ii_graph_valid ||
            engine->d_scheme_ii_graph_m != m ||
            engine->d_scheme_ii_graph_n != n ||
            engine->d_scheme_ii_graph_k != k ||
            engine->d_scheme_ii_graph_num_moduli != num_moduli ||
            engine->d_scheme_ii_graph_A != d_A ||
            engine->d_scheme_ii_graph_B != d_B ||
            engine->d_scheme_ii_graph_C != d_C;

        if (need_capture) {
            destroy_scheme_ii_graph(engine);
            cudaStreamSynchronize(stream);
            cudaError_t cap_st = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (cap_st == cudaSuccess) {
                launch_fused_pipeline(handle, stream, d_A, d_B, d_C, m, n, k, m_pad, n_pad, k_pad,
                                      num_moduli, S, M, M_half, d_primes, d_inv_table, d_P,
                                      d_eA, d_eB, d_A_mod, d_B_mod, d_C_mod, nullptr, false);
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
        launch_fused_pipeline(handle, stream, d_A, d_B, d_C, m, n, k, m_pad, n_pad, k_pad,
                              num_moduli, S, M, M_half, d_primes, d_inv_table, d_P,
                              d_eA, d_eB, d_A_mod, d_B_mod, d_C_mod, ev, profile);
    }

    if (profile) {
        cudaEventSynchronize(ev[6]);
        contract.stages.clear();
        if (used_graph) {
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

    contract.used_scheme_ii_fused = true;
    contract.used_true_fused_kernel = false;
    contract.used_cuda_graph = used_graph;
    contract.num_moduli = num_moduli;
    if (contract.fallback_mode.empty()) contract.fallback_mode = "None";
}

} // namespace ozaki
