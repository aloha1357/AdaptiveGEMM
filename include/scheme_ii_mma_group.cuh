#pragma once

#include "mma_sm89.cuh"
#include "scheme_ii_int192.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace ozaki {

__device__ __forceinline__ int scheme_ii_reduce_mod(int acc, int p) {
    int r = acc % p;
    if (r < 0) r += p;
    return r;
}

__device__ __forceinline__ int scheme_ii_modinv(int a, int m) {
    int m0 = m, y = 0, x = 1;
    a %= m;
    if (a < 0) a += m;
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

__device__ __forceinline__ int8_t scheme_ii_quantize_residue(double val, int exp, int S, int p) {
    double scaled = val * ldexp(1.0, S - exp);
    int64_t v_int = llround(scaled);
    int rem = (int)(v_int % p);
    if (rem < 0) rem += p;
    if (rem > p / 2) rem -= p;
    return (int8_t)rem;
}

__device__ __forceinline__ void scheme_ii_garner_inplace(
    int* v, const int32_t* primes, int n, const int32_t* inv_table, int p_base) {
    for (int i = 0; i < n - 1; ++i) {
        int v_ii = v[i];
        for (int j = i + 1; j < n; ++j) {
            int p = primes[p_base + j];
            long long diff = (long long)v[j] - v_ii;
            diff %= p;
            if (diff < 0) diff += p;
            int inv = inv_table[(p_base + i) * 20 + (p_base + j)];
            v[j] = (int)((diff * (long long)inv) % p);
        }
    }
}

__device__ __forceinline__ uint64_t scheme_ii_mixradix_u64(
    const int* v, const int32_t* primes, int n, int p_base) {
    uint64_t X = 0;
    uint64_t P = 1;
    for (int i = 0; i < n; ++i) {
        X += (uint64_t)(uint32_t)v[i] * P;
        P *= (uint32_t)primes[p_base + i];
    }
    return X;
}

__device__ __forceinline__ void scheme_ii_int192_sub(Int192& out, const Int192& a, const Int192& b) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        long long d = (long long)a.limbs[i] - (long long)borrow - (long long)b.limbs[i];
        if (d < 0) {
            d += 4294967296LL;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.limbs[i] = (uint32_t)d;
    }
}

__device__ __forceinline__ double scheme_ii_crt_to_double(
    const uint32_t* v, int n, const Int192* P, Int192 M, Int192 M_half) {
    Int192 X;
    for (int i = 0; i < n; ++i) {
        X.add_mul_scalar(P[i], v[i]);
    }
    bool is_neg = X.is_greater_than(M_half);
    if (is_neg) {
        Int192 diff;
        scheme_ii_int192_sub(diff, M, X);
        X = diff;
    }
    double out = X.to_double();
    if (is_neg) out = -out;
    return out;
}

__device__ __forceinline__ double scheme_ii_pow2i_fast(int e) {
    int biased = e + 1023;
    if (biased > 0 && biased < 2047) {
        return __longlong_as_double((unsigned long long)biased << 52);
    }
    return ldexp(1.0, e);
}

__device__ __forceinline__ void scheme_ii_cta_swizzle(int swizzle, int& bx, int& by, int grid_n, int grid_m) {
    if (swizzle == 1) {
        const int mask = 3;
        if ((grid_n & mask) == 0) {
            bx = (bx & ~mask) | ((bx ^ by) & mask);
        }
    } else if (swizzle == 2) {
        int bid = by * grid_n + bx;
        by = bid % grid_m;
        bx = bid / grid_m;
    }
}

// Sequential grouped INT8 MMA. One launch covers GroupSize moduli.
// PartialCRT=false: write C_mod for each modulus.
// PartialCRT=true: online Garner of the group, write one uint32 plane (C mod M_g).
// Tile-local A/B residue materialization is a separate prologue kernel.
// --use_true_fused_kernel with group_size < num_moduli selects this partial CRT
// path (M8.3), not the all-moduli true fused kernel (M8.4).
template <int GroupSize, bool PartialCRT, int Stages, int BM, int BN, int Swizzle, int MinBlocks>
__global__ void __launch_bounds__(256, MinBlocks) scheme_ii_grouped_mma_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int m, int n, int k,
    int n_group,
    const int32_t* __restrict__ primes,
    const int32_t* __restrict__ inv_table,
    int group_base)
{
    constexpr int BK = 64;

    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int wm = wid / 2;
    int wn = wid % 2;

    int bx = (int)blockIdx.x;
    int by = (int)blockIdx.y;
    scheme_ii_cta_swizzle(Swizzle, bx, by, (int)gridDim.x, (int)gridDim.y);
    int bm_off = by * BM;
    int bn_off = bx * BN;

    extern __shared__ int8_t smem[];
    int8_t* sA_ptr = smem;
    int8_t* sB_ptr = smem + Stages * BM * 128;

    auto swizzle_A = [](int row, int col) {
        return col ^ ((row & 7) << 4);
    };
    auto swizzle_B = [](int row, int col) {
        return col ^ (((row & 3) ^ ((row >> 4) & 3)) << 4);
    };

    [[maybe_unused]] uint32_t partial_x[PartialCRT ? (BM / 64) : 1][PartialCRT ? (BN / 16) : 1][PartialCRT ? 4 : 1];
    [[maybe_unused]] uint32_t partial_M = 1;
    if constexpr (PartialCRT) {
        #pragma unroll
        for (int i = 0; i < BM / 64; ++i)
            #pragma unroll
            for (int j = 0; j < BN / 16; ++j)
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    partial_x[i][j][t] = 0;
    }

    for (int g = 0; g < n_group; ++g) {
        const int8_t* A_g = A + (size_t)g * m * k;
        const int8_t* B_g = B + (size_t)g * k * n;

        uint32_t a_frag[2][BM / 64][4];
        uint32_t b_frag[2][BN / 16][2];
        int32_t acc[BM / 64][BN / 16][4];
        #pragma unroll
        for (int i = 0; i < BM / 64; ++i)
            #pragma unroll
            for (int j = 0; j < BN / 16; ++j)
                #pragma unroll
                for (int t = 0; t < 4; ++t) acc[i][j][t] = 0;

        auto load_a_smem = [&](int buf, int k_offset) {
            int8_t* sA = sA_ptr + buf * BM * 128;
            for (int i = tid; i < BM * BK / 16; i += 256) {
                int row = (i * 16) / BK;
                int col = (i * 16) % BK;
                int gm = bm_off + row;
                int gk = k_offset + col;
                if (gm < m && gk < k) {
                    cp_async_16(&sA[row * 128 + swizzle_A(row, col)], &A_g[(size_t)gm * k + gk]);
                }
            }
        };
        auto load_b_smem = [&](int buf, int k_offset) {
            int8_t* sB = sB_ptr + buf * BN * 64;
            for (int i = tid; i < BK * BN / 16; i += 256) {
                int row = (i * 16) / BN;
                int col = (i * 16) % BN;
                int gk = k_offset + row;
                int gn = bn_off + col;
                if (gk < k && gn < n) {
                    int4 v = *(int4*)&B_g[(size_t)gk * n + gn];
                    int8_t* bytes = (int8_t*)&v;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        sB[(col + j) * 64 + swizzle_B(col + j, row)] = bytes[j];
                    }
                }
            }
        };
        auto load_a_frag = [&](int buf, int k_step, int fbuf) {
            int8_t* sA = sA_ptr + buf * BM * 128;
            int ms = wm * (BM / 4);
            #pragma unroll
            for (int i = 0; i < BM / 64; i++) {
                int a_row = ms + i * 16 + (lane % 16);
                int a_col = k_step + (lane / 16) * 16;
                uint32_t addr = __cvta_generic_to_shared(&sA[a_row * 128 + swizzle_A(a_row, a_col)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a_frag[fbuf][i][0]), "=r"(a_frag[fbuf][i][1]),
                      "=r"(a_frag[fbuf][i][2]), "=r"(a_frag[fbuf][i][3])
                    : "r"(addr));
            }
        };
        auto load_b_frag = [&](int buf, int k_step, int fbuf) {
            int8_t* sB = sB_ptr + buf * BN * 64;
            int ns = wn * (BN / 2);
            #pragma unroll
            for (int j = 0; j < BN / 16; j++) {
                int b_row = ns + j * 8 + (lane % 8);
                int b_col = k_step + (lane / 8 % 2) * 16;
                uint32_t addr = __cvta_generic_to_shared(&sB[b_row * 64 + swizzle_B(b_row, b_col)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(b_frag[fbuf][j][0]), "=r"(b_frag[fbuf][j][1])
                    : "r"(addr));
            }
        };

        int prefetch_k = 0;
        for (int p = 0; p < Stages - 1; ++p) {
            if (prefetch_k < k) {
                load_a_smem(p, prefetch_k);
                load_b_smem(p, prefetch_k);
                cp_async_commit();
                prefetch_k += BK;
            }
        }
        cp_async_wait<0>();
        __syncthreads();

        int read_stage = 0;
        int k1 = 0;
        if (0 < k) {
            load_a_frag(read_stage, k1, 0);
            load_b_frag(read_stage, k1, 0);
        }

        for (int ck = 0; ck < k; ck += 32) {
            int p = (ck / 32) % 2;
            int next_k1 = k1 + 32;
            int next_stage = read_stage;
            bool fetch_next_stage = false;
            if (next_k1 >= BK) {
                next_k1 = 0;
                next_stage = (read_stage + 1);
                if (next_stage >= Stages) next_stage = 0;
                fetch_next_stage = true;
            }
            bool do_prefetch = (ck + 32 < k);
            if (fetch_next_stage && prefetch_k < k) {
                int write_stage = (next_stage + Stages - 2) % Stages;
                load_a_smem(write_stage, prefetch_k);
                load_b_smem(write_stage, prefetch_k);
                cp_async_commit();
                prefetch_k += BK;
            }
            if (next_k1 == 0 && do_prefetch) {
                cp_async_wait<Stages - 2>();
                __syncthreads();
            }
            if (do_prefetch) {
                load_a_frag(next_stage, next_k1, p ^ 1);
                load_b_frag(next_stage, next_k1, p ^ 1);
            }
            #pragma unroll
            for (int i = 0; i < BM / 64; i++) {
                #pragma unroll
                for (int j = 0; j < BN / 16; j++) {
                    mma_m16n8k32_s8(acc[i][j], a_frag[p][i], b_frag[p][j], acc[i][j]);
                }
            }
            k1 = next_k1;
            read_stage = next_stage;
        }

        if constexpr (PartialCRT) {
            int pmod = primes[group_base + g];
            int inv_M = 1;
            if (g > 0) {
                inv_M = scheme_ii_modinv((int)(partial_M % (uint32_t)pmod), pmod);
            }
            #pragma unroll
            for (int i = 0; i < BM / 64; ++i)
                #pragma unroll
                for (int j = 0; j < BN / 16; ++j)
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        int r_mod = scheme_ii_reduce_mod(acc[i][j][t], pmod);
                        if (g == 0) {
                            partial_x[i][j][t] = (uint32_t)r_mod;
                        } else {
                            int x_mod = (int)(partial_x[i][j][t] % (uint32_t)pmod);
                            int diff = r_mod - x_mod;
                            diff %= pmod;
                            if (diff < 0) diff += pmod;
                            uint32_t digit = (uint32_t)(((long long)diff * inv_M) % pmod);
                            partial_x[i][j][t] += (uint32_t)((uint64_t)digit * partial_M);
                        }
                    }
            partial_M *= (uint32_t)pmod;
            (void)inv_table;
        } else {
            const int r_off[2] = { lane / 4, lane / 4 + 8 };
            const int c_off = (lane % 4) * 2;
            int32_t* C_g = C + (size_t)g * m * n;
            #pragma unroll
            for (int i = 0; i < BM / 64; ++i) {
                #pragma unroll
                for (int j = 0; j < BN / 16; ++j) {
                    #pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        int gm = bm_off + wm * (BM / 4) + i * 16 + r_off[r];
                        int gn = bn_off + wn * (BN / 2) + j * 8 + c_off;
                        if (gm < m && gn < n) {
                            int2 v;
                            v.x = acc[i][j][r * 2];
                            v.y = acc[i][j][r * 2 + 1];
                            *(int2*)&C_g[(size_t)gm * n + gn] = v;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (PartialCRT) {
        const int r_off[2] = { lane / 4, lane / 4 + 8 };
        const int c_off = (lane % 4) * 2;
        #pragma unroll
        for (int i = 0; i < BM / 64; ++i) {
            #pragma unroll
            for (int j = 0; j < BN / 16; ++j) {
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    int gm = bm_off + wm * (BM / 4) + i * 16 + r_off[r];
                    int gn = bn_off + wn * (BN / 2) + j * 8 + c_off;
                    if (gm < m && gn < n) {
                        #pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            if (gn + c >= n) break;
                            uint32_t* C_u32 = reinterpret_cast<uint32_t*>(C);
                            C_u32[(size_t)gm * n + gn + c] = partial_x[i][j][r * 2 + c];
                        }
                    }
                }
            }
        }
    }
}

// Tile-local prologue fusion: CTA reads double A/B, quantizes the active
// modulus into smem, then runs the same grouped MMA + online partial CRT.
// Opt-in only; A/B residue planes are not written to global memory.
template <int GroupSize, int Stages, int BM, int BN, int Swizzle>
__global__ void __launch_bounds__(256, 1) scheme_ii_grouped_mma_prologue_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    uint32_t* __restrict__ C_partial,
    const int* __restrict__ eA,
    const int* __restrict__ eB,
    int m, int n, int k,
    int m_pad, int n_pad, int k_pad,
    int n_group, int S,
    const int32_t* __restrict__ primes,
    int group_base)
{
    constexpr int BK = 64;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int wm = wid / 2;
    int wn = wid % 2;

    int bx = (int)blockIdx.x;
    int by = (int)blockIdx.y;
    scheme_ii_cta_swizzle(Swizzle, bx, by, (int)gridDim.x, (int)gridDim.y);
    int bm_off = by * BM;
    int bn_off = bx * BN;

    extern __shared__ int8_t smem[];
    int8_t* sA_ptr = smem;
    int8_t* sB_ptr = smem + Stages * BM * 128;

    auto swizzle_A = [](int row, int col) {
        return col ^ ((row & 7) << 4);
    };
    auto swizzle_B = [](int row, int col) {
        return col ^ (((row & 3) ^ ((row >> 4) & 3)) << 4);
    };

    uint32_t partial_x[BM / 64][BN / 16][4];
    uint32_t partial_M = 1;
    #pragma unroll
    for (int i = 0; i < BM / 64; ++i)
        #pragma unroll
        for (int j = 0; j < BN / 16; ++j)
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                partial_x[i][j][t] = 0;

    for (int g = 0; g < n_group; ++g) {
        int pmod = primes[group_base + g];
        uint32_t a_frag[2][BM / 64][4];
        uint32_t b_frag[2][BN / 16][2];
        int32_t acc[BM / 64][BN / 16][4];
        #pragma unroll
        for (int i = 0; i < BM / 64; ++i)
            #pragma unroll
            for (int j = 0; j < BN / 16; ++j)
                #pragma unroll
                for (int t = 0; t < 4; ++t) acc[i][j][t] = 0;

        auto quant_a_smem = [&](int buf, int k_offset) {
            int8_t* sA = sA_ptr + buf * BM * 128;
            for (int i = tid; i < BM * BK; i += 256) {
                int row = i / BK;
                int col = i % BK;
                int gm = bm_off + row;
                int gk = k_offset + col;
                int8_t q = 0;
                if (gm < m && gk < k) {
                    q = scheme_ii_quantize_residue(A[(size_t)gm * k + gk], eA[gm], S, pmod);
                }
                sA[row * 128 + swizzle_A(row, col)] = q;
            }
        };
        auto quant_b_smem = [&](int buf, int k_offset) {
            int8_t* sB = sB_ptr + buf * BN * 64;
            for (int i = tid; i < BK * BN; i += 256) {
                int row = i / BN;
                int col = i % BN;
                int gk = k_offset + row;
                int gn = bn_off + col;
                int8_t q = 0;
                if (gk < k && gn < n) {
                    q = scheme_ii_quantize_residue(B[(size_t)gk * n + gn], eB[gn], S, pmod);
                }
                sB[col * 64 + swizzle_B(col, row)] = q;
            }
        };
        auto load_a_frag = [&](int buf, int k_step, int fbuf) {
            int8_t* sA = sA_ptr + buf * BM * 128;
            int ms = wm * (BM / 4);
            #pragma unroll
            for (int i = 0; i < BM / 64; i++) {
                int a_row = ms + i * 16 + (lane % 16);
                int a_col = k_step + (lane / 16) * 16;
                uint32_t addr = __cvta_generic_to_shared(&sA[a_row * 128 + swizzle_A(a_row, a_col)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a_frag[fbuf][i][0]), "=r"(a_frag[fbuf][i][1]),
                      "=r"(a_frag[fbuf][i][2]), "=r"(a_frag[fbuf][i][3])
                    : "r"(addr));
            }
        };
        auto load_b_frag = [&](int buf, int k_step, int fbuf) {
            int8_t* sB = sB_ptr + buf * BN * 64;
            int ns = wn * (BN / 2);
            #pragma unroll
            for (int j = 0; j < BN / 16; j++) {
                int b_row = ns + j * 8 + (lane % 8);
                int b_col = k_step + (lane / 8 % 2) * 16;
                uint32_t addr = __cvta_generic_to_shared(&sB[b_row * 64 + swizzle_B(b_row, b_col)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(b_frag[fbuf][j][0]), "=r"(b_frag[fbuf][j][1])
                    : "r"(addr));
            }
        };

        int prefetch_k = 0;
        for (int p = 0; p < Stages - 1; ++p) {
            if (prefetch_k < k_pad) {
                quant_a_smem(p, prefetch_k);
                quant_b_smem(p, prefetch_k);
                prefetch_k += BK;
            }
        }
        __syncthreads();

        int read_stage = 0;
        int k1 = 0;
        if (0 < k_pad) {
            load_a_frag(read_stage, k1, 0);
            load_b_frag(read_stage, k1, 0);
        }

        for (int ck = 0; ck < k_pad; ck += 32) {
            int p = (ck / 32) % 2;
            int next_k1 = k1 + 32;
            int next_stage = read_stage;
            bool fetch_next_stage = false;
            if (next_k1 >= BK) {
                next_k1 = 0;
                next_stage = (read_stage + 1);
                if (next_stage >= Stages) next_stage = 0;
                fetch_next_stage = true;
            }
            bool do_prefetch = (ck + 32 < k_pad);
            if (fetch_next_stage && prefetch_k < k_pad) {
                int write_stage = (next_stage + Stages - 2) % Stages;
                quant_a_smem(write_stage, prefetch_k);
                quant_b_smem(write_stage, prefetch_k);
                prefetch_k += BK;
            }
            if (next_k1 == 0 && do_prefetch) {
                __syncthreads();
            }
            if (do_prefetch) {
                load_a_frag(next_stage, next_k1, p ^ 1);
                load_b_frag(next_stage, next_k1, p ^ 1);
            }
            #pragma unroll
            for (int i = 0; i < BM / 64; i++) {
                #pragma unroll
                for (int j = 0; j < BN / 16; j++) {
                    mma_m16n8k32_s8(acc[i][j], a_frag[p][i], b_frag[p][j], acc[i][j]);
                }
            }
            k1 = next_k1;
            read_stage = next_stage;
        }

        int inv_M = 1;
        if (g > 0) {
            inv_M = scheme_ii_modinv((int)(partial_M % (uint32_t)pmod), pmod);
        }
        #pragma unroll
        for (int i = 0; i < BM / 64; ++i)
            #pragma unroll
            for (int j = 0; j < BN / 16; ++j)
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int r_mod = scheme_ii_reduce_mod(acc[i][j][t], pmod);
                    if (g == 0) {
                        partial_x[i][j][t] = (uint32_t)r_mod;
                    } else {
                        int x_mod = (int)(partial_x[i][j][t] % (uint32_t)pmod);
                        int diff = r_mod - x_mod;
                        diff %= pmod;
                        if (diff < 0) diff += pmod;
                        uint32_t digit = (uint32_t)(((long long)diff * inv_M) % pmod);
                        partial_x[i][j][t] += (uint32_t)((uint64_t)digit * partial_M);
                    }
                }
        partial_M *= (uint32_t)pmod;
        __syncthreads();
    }

    const int r_off[2] = { lane / 4, lane / 4 + 8 };
    const int c_off = (lane % 4) * 2;
    #pragma unroll
    for (int i = 0; i < BM / 64; ++i) {
        #pragma unroll
        for (int j = 0; j < BN / 16; ++j) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                int gm = bm_off + wm * (BM / 4) + i * 16 + r_off[r];
                int gn = bn_off + wn * (BN / 2) + j * 8 + c_off;
                if (gm < m && gn < n) {
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        if (gn + c >= n) break;
                        C_partial[(size_t)gm * n_pad + gn + c] = partial_x[i][j][r * 2 + c];
                    }
                }
            }
        }
    }
    (void)m_pad;
}

// Small-tile true fused kernel: 64x16 output tile, 4 values/thread, all moduli,
// write double C. NCU shows this path is FP64/ALU epilogue bound, not HBM bound.
// M8.4 is a control path. Do not treat it as the speed path unless it beats M8.3.
__global__ void scheme_ii_true_fused_kernel(
    const int8_t* __restrict__ A_mods,
    const int8_t* __restrict__ B_mods,
    double* __restrict__ C,
    const int* __restrict__ eA,
    const int* __restrict__ eB,
    int m, int n, int k,
    int m_pad, int n_pad, int k_pad,
    int num_moduli, int S,
    const int32_t* __restrict__ primes,
    const int32_t* __restrict__ inv_table,
    Int192 M, Int192 M_half,
    const Int192* __restrict__ P)
{
    constexpr int BM = 64;
    constexpr int BN = 16;
    constexpr int BK = 64;

    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int wm = wid / 2;
    int wn = wid % 2;

    int bm_off = blockIdx.y * BM;
    int bn_off = blockIdx.x * BN;
    if (bm_off >= m || bn_off >= n) return;

    extern __shared__ int8_t smem[];
    int8_t* sA = smem;
    int8_t* sB = smem + BM * 128;

    auto swizzle_A = [](int row, int col) {
        return col ^ ((row & 7) << 4);
    };
    auto swizzle_B = [](int row, int col) {
        return col ^ (((row & 3) ^ ((row >> 4) & 3)) << 4);
    };

    int residues[20][4];

    for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
        const int8_t* A = A_mods + (size_t)p_idx * m_pad * k_pad;
        const int8_t* B = B_mods + (size_t)p_idx * k_pad * n_pad;
        int32_t acc[4] = {0, 0, 0, 0};

        for (int ko = 0; ko < k_pad; ko += BK) {
            for (int i = tid; i < BM * BK / 16; i += 256) {
                int row = (i * 16) / BK;
                int col = (i * 16) % BK;
                int gm = bm_off + row;
                int gk = ko + col;
                if (gm < m_pad && gk < k_pad) {
                    cp_async_16(&sA[row * 128 + swizzle_A(row, col)], &A[(size_t)gm * k_pad + gk]);
                }
            }
            for (int i = tid; i < BK * BN / 16; i += 256) {
                int row = (i * 16) / BN;
                int col = (i * 16) % BN;
                int gk = ko + row;
                int gn = bn_off + col;
                if (gk < k_pad && gn < n_pad) {
                    int4 v = *(int4*)&B[(size_t)gk * n_pad + gn];
                    int8_t* bytes = (int8_t*)&v;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        sB[(col + j) * 64 + swizzle_B(col + j, row)] = bytes[j];
                    }
                }
            }
            cp_async_commit();
            cp_async_wait<0>();
            __syncthreads();

            for (int k1 = 0; k1 < BK; k1 += 32) {
                uint32_t a_frag[4], b_frag[2];
                int a_row = wm * 16 + (lane % 16);
                int a_col = k1 + (lane / 16) * 16;
                uint32_t addr_a = __cvta_generic_to_shared(&sA[a_row * 128 + swizzle_A(a_row, a_col)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(addr_a));
                int b_row = wn * 8 + (lane % 8);
                int b_col = k1 + (lane / 8 % 2) * 16;
                uint32_t addr_b = __cvta_generic_to_shared(&sB[b_row * 64 + swizzle_B(b_row, b_col)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(b_frag[0]), "=r"(b_frag[1])
                    : "r"(addr_b));
                mma_m16n8k32_s8(acc, a_frag, b_frag, acc);
            }
            __syncthreads();
        }

        int p = primes[p_idx];
        #pragma unroll
        for (int t = 0; t < 4; ++t) residues[p_idx][t] = scheme_ii_reduce_mod(acc[t], p);
    }

    const int r_off[2] = { lane / 4, lane / 4 + 8 };
    const int c_off = (lane % 4) * 2;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            int gm = bm_off + wm * 16 + r_off[r];
            int gn = bn_off + wn * 8 + c_off + c;
            if (gm < m && gn < n) {
                int vg[20];
                uint32_t vu[20];
                for (int p_idx = 0; p_idx < num_moduli; ++p_idx) vg[p_idx] = residues[p_idx][r * 2 + c];
                scheme_ii_garner_inplace(vg, primes, num_moduli, inv_table, 0);
                for (int p_idx = 0; p_idx < num_moduli; ++p_idx) vu[p_idx] = (uint32_t)vg[p_idx];
                double C_int = scheme_ii_crt_to_double(vu, num_moduli, P, M, M_half);
                int exp_sum = 2 * S - eA[gm] - eB[gn];
                C[(size_t)gm * n + gn] = C_int * scheme_ii_pow2i_fast(-exp_sum);
            }
        }
    }
}

__global__ void scheme_ii_combine_partial_crt_kernel(
    const uint32_t* __restrict__ C_partial,
    double* __restrict__ C,
    const int* __restrict__ eA,
    const int* __restrict__ eB,
    int m, int n, int m_pad, int n_pad,
    int num_groups, int S,
    const uint32_t* __restrict__ group_moduli,
    const int32_t* __restrict__ inv_table,
    Int192 M, Int192 M_half,
    const Int192* __restrict__ P)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;

    uint32_t v[20];
    for (int g = 0; g < num_groups; ++g) {
        uint32_t Mg = group_moduli[g];
        uint32_t r = C_partial[(size_t)g * m_pad * n_pad + i * n_pad + j];
        r %= Mg;
        v[g] = r;
    }

    for (int g = 0; g < num_groups - 1; ++g) {
        uint32_t v_ii = v[g];
        for (int h = g + 1; h < num_groups; ++h) {
            uint32_t Mg = group_moduli[h];
            long long diff = (long long)v[h] - (long long)v_ii;
            diff %= (long long)Mg;
            if (diff < 0) diff += Mg;
            uint32_t inv = (uint32_t)inv_table[g * 20 + h];
            v[h] = (uint32_t)(((unsigned long long)diff * inv) % Mg);
        }
    }

    double C_int = scheme_ii_crt_to_double(v, num_groups, P, M, M_half);
    int exp_sum = 2 * S - eA[i] - eB[j];
    C[(size_t)i * n + j] = C_int * scheme_ii_pow2i_fast(-exp_sum);
}

template <int NumGroups, int VecN>
__global__ void scheme_ii_combine_partial_crt_fixed_kernel(
    const uint32_t* __restrict__ C_partial,
    double* __restrict__ C,
    const int* __restrict__ eA,
    const int* __restrict__ eB,
    int m, int n, int m_pad, int n_pad,
    int S,
    const uint32_t* __restrict__ group_moduli,
    const int32_t* __restrict__ inv_table,
    Int192 M, Int192 M_half,
    const Int192* __restrict__ P)
{
    int j0 = (blockIdx.x * blockDim.x + threadIdx.x) * VecN;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j0 >= n) return;

    uint32_t Mg_reg[NumGroups];
    uint32_t inv_reg[NumGroups][NumGroups];
    #pragma unroll
    for (int g = 0; g < NumGroups; ++g) {
        Mg_reg[g] = group_moduli[g];
        #pragma unroll
        for (int h = 0; h < NumGroups; ++h) inv_reg[g][h] = 0;
        #pragma unroll
        for (int h = g + 1; h < NumGroups; ++h) {
            inv_reg[g][h] = (uint32_t)inv_table[g * 20 + h];
        }
    }
    int eA_i = eA[i];

    #pragma unroll
    for (int c = 0; c < VecN; ++c) {
        int j = j0 + c;
        if (j >= n) continue;
        uint32_t v[NumGroups];
        #pragma unroll
        for (int g = 0; g < NumGroups; ++g) {
            uint32_t r = C_partial[(size_t)g * m_pad * n_pad + i * n_pad + j];
            v[g] = r % Mg_reg[g];
        }
        #pragma unroll
        for (int g = 0; g < NumGroups - 1; ++g) {
            uint32_t v_ii = v[g];
            #pragma unroll
            for (int h = g + 1; h < NumGroups; ++h) {
                uint32_t Mg = Mg_reg[h];
                long long diff = (long long)v[h] - (long long)v_ii;
                diff %= (long long)Mg;
                if (diff < 0) diff += Mg;
                v[h] = (uint32_t)(((unsigned long long)diff * inv_reg[g][h]) % Mg);
            }
        }
        double C_int = scheme_ii_crt_to_double(v, NumGroups, P, M, M_half);
        int exp_sum = 2 * S - eA_i - eB[j];
        C[(size_t)i * n + j] = C_int * scheme_ii_pow2i_fast(-exp_sum);
    }
}

enum class SchemeIIMMAVariant : int {
    Default = 0,
    Tile128x128_S2 = 1,
    Tile128x128_S2_NXor = 2,
    Tile128x64_S2_NXor = 3,
    Tile64x128_S2_NXor = 4,
    Tile128x128_S3_NXor = 5,
    Tile128x128_S2_MMajor = 6
};

// Default experimental speed path. Updated after autotune measurements.
inline int scheme_ii_default_mma_variant() {
    return (int)SchemeIIMMAVariant::Tile128x128_S2;
}

inline void scheme_ii_resolve_mma_variant(int variant, int& bm, int& bn, int& stages, int& swizzle) {
    if (variant <= 0) variant = scheme_ii_default_mma_variant();
    switch (variant) {
        case (int)SchemeIIMMAVariant::Tile128x128_S2:
            bm = 128; bn = 128; stages = 2; swizzle = 0; break;
        case (int)SchemeIIMMAVariant::Tile128x128_S2_NXor:
            bm = 128; bn = 128; stages = 2; swizzle = 1; break;
        case (int)SchemeIIMMAVariant::Tile128x64_S2_NXor:
            bm = 128; bn = 64; stages = 2; swizzle = 1; break;
        case (int)SchemeIIMMAVariant::Tile64x128_S2_NXor:
            bm = 64; bn = 128; stages = 2; swizzle = 1; break;
        case (int)SchemeIIMMAVariant::Tile128x128_S3_NXor:
            bm = 128; bn = 128; stages = 3; swizzle = 1; break;
        case (int)SchemeIIMMAVariant::Tile128x128_S2_MMajor:
            bm = 128; bn = 128; stages = 2; swizzle = 2; break;
        default:
            bm = 128; bn = 128; stages = 2; swizzle = 0; break;
    }
}

template <int GroupSize, bool PartialCRT, int Stages, int BM, int BN, int Swizzle, int MinBlocks>
inline cudaError_t scheme_ii_launch_grouped_mma_inst(
    const int8_t* A, const int8_t* B, int32_t* C,
    int m, int n, int k, int n_group,
    const int32_t* primes, const int32_t* inv_table, int group_base,
    cudaStream_t stream)
{
    auto* fn = scheme_ii_grouped_mma_kernel<GroupSize, PartialCRT, Stages, BM, BN, Swizzle, MinBlocks>;
    int smem_size = Stages * BM * 128 + Stages * BN * 64;
    dim3 block(256);
    dim3 grid(n / BN, m / BM);
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    fn<<<grid, block, smem_size, stream>>>(
        A, B, C, m, n, k, n_group, primes, inv_table, group_base);
    return cudaGetLastError();
}

inline cudaError_t scheme_ii_launch_grouped_mma(
    const int8_t* A, const int8_t* B, int32_t* C,
    int m, int n, int k, int n_group, int group_size,
    const int32_t* primes, const int32_t* inv_table, int group_base,
    bool partial_crt, cudaStream_t stream, int variant = 0)
{
    if (m % 64 != 0 || n % 64 != 0 || k % 64 != 0) return cudaErrorInvalidValue;
    int bm = 128, bn = 128, stages = 2, swizzle = 0;
    if (partial_crt) {
        scheme_ii_resolve_mma_variant(variant, bm, bn, stages, swizzle);
    } else {
        bm = 128; bn = 128; stages = 3; swizzle = 0;
    }
    if (m % bm != 0 || n % bn != 0) return cudaErrorInvalidValue;

    if (partial_crt) {
        if (group_size >= 4) {
            if (bm == 128 && bn == 64 && stages == 2 && swizzle == 1)
                return scheme_ii_launch_grouped_mma_inst<4, true, 2, 128, 64, 1, 2>(
                    A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
            if (bm == 64 && bn == 128 && stages == 2 && swizzle == 1)
                return scheme_ii_launch_grouped_mma_inst<4, true, 2, 64, 128, 1, 2>(
                    A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
            if (bm == 128 && bn == 128 && stages == 3 && swizzle == 1)
                return scheme_ii_launch_grouped_mma_inst<4, true, 3, 128, 128, 1, 1>(
                    A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
            if (bm == 128 && bn == 128 && stages == 2 && swizzle == 1)
                return scheme_ii_launch_grouped_mma_inst<4, true, 2, 128, 128, 1, 1>(
                    A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
            if (bm == 128 && bn == 128 && stages == 2 && swizzle == 2)
                return scheme_ii_launch_grouped_mma_inst<4, true, 2, 128, 128, 2, 1>(
                    A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
            return scheme_ii_launch_grouped_mma_inst<4, true, 2, 128, 128, 0, 1>(
                A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
        }
        if (bm == 128 && bn == 64 && stages == 2 && swizzle == 1)
            return scheme_ii_launch_grouped_mma_inst<2, true, 2, 128, 64, 1, 2>(
                A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
        if (bm == 128 && bn == 128 && stages == 2 && swizzle == 1)
            return scheme_ii_launch_grouped_mma_inst<2, true, 2, 128, 128, 1, 1>(
                A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
        return scheme_ii_launch_grouped_mma_inst<2, true, 2, 128, 128, 0, 1>(
            A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
    }

    if (group_size >= 4) {
        return scheme_ii_launch_grouped_mma_inst<4, false, 3, 128, 128, 0, 1>(
            A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
    }
    return scheme_ii_launch_grouped_mma_inst<2, false, 3, 128, 128, 0, 1>(
        A, B, C, m, n, k, n_group, primes, inv_table, group_base, stream);
}

inline cudaError_t scheme_ii_launch_grouped_mma_prologue(
    const double* A, const double* B, uint32_t* C_partial,
    const int* eA, const int* eB,
    int m, int n, int k, int m_pad, int n_pad, int k_pad,
    int n_group, int S,
    const int32_t* primes, int group_base,
    cudaStream_t stream, int variant = 0)
{
    if (m_pad % 64 != 0 || n_pad % 64 != 0 || k_pad % 64 != 0) return cudaErrorInvalidValue;
    int bm = 128, bn = 128, stages = 2, swizzle = 0;
    scheme_ii_resolve_mma_variant(variant, bm, bn, stages, swizzle);
    if (m_pad % bm != 0 || n_pad % bn != 0) return cudaErrorInvalidValue;
    int smem_size = stages * bm * 128 + stages * bn * 64;
    dim3 block(256);
    dim3 grid(n_pad / bn, m_pad / bm);

    if (bm == 128 && bn == 64) {
        auto* fn = scheme_ii_grouped_mma_prologue_kernel<4, 2, 128, 64, 1>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        fn<<<grid, block, smem_size, stream>>>(
            A, B, C_partial, eA, eB, m, n, k, m_pad, n_pad, k_pad,
            n_group, S, primes, group_base);
    } else {
        auto* fn = scheme_ii_grouped_mma_prologue_kernel<4, 2, 128, 128, 1>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        fn<<<grid, block, smem_size, stream>>>(
            A, B, C_partial, eA, eB, m, n, k, m_pad, n_pad, k_pad,
            n_group, S, primes, group_base);
    }
    return cudaGetLastError();
}

inline cudaError_t scheme_ii_launch_true_fused(
    const int8_t* A_mods, const int8_t* B_mods, double* C,
    const int* eA, const int* eB,
    int m, int n, int k, int m_pad, int n_pad, int k_pad,
    int num_moduli, int S,
    const int32_t* primes, const int32_t* inv_table,
    Int192 M, Int192 M_half, const Int192* P,
    cudaStream_t stream)
{
    if (m_pad % 64 != 0 || n_pad % 16 != 0 || k_pad % 64 != 0) return cudaErrorInvalidValue;
    constexpr int BM = 64, BN = 16;
    dim3 block(256);
    dim3 grid(n_pad / BN, m_pad / BM);
    int smem_size = BM * 128 + BN * 64;
    cudaFuncSetAttribute(scheme_ii_true_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    scheme_ii_true_fused_kernel<<<grid, block, smem_size, stream>>>(
        A_mods, B_mods, C, eA, eB, m, n, k, m_pad, n_pad, k_pad,
        num_moduli, S, primes, inv_table, M, M_half, P);
    return cudaGetLastError();
}

inline void scheme_ii_launch_combine_partial_crt(
    const uint32_t* C_partial, double* C,
    const int* eA, const int* eB,
    int m, int n, int m_pad, int n_pad,
    int num_groups, int S,
    const uint32_t* group_moduli, const int32_t* inv_table,
    Int192 M, Int192 M_half, const Int192* P,
    cudaStream_t stream)
{
    dim3 block2d(16, 16);
    dim3 gridC((n + 15) / 16, (m + 15) / 16);
    if (num_groups == 4) {
        scheme_ii_combine_partial_crt_fixed_kernel<4, 1><<<gridC, block2d, 0, stream>>>(
            C_partial, C, eA, eB, m, n, m_pad, n_pad, S,
            group_moduli, inv_table, M, M_half, P);
    } else if (num_groups == 8) {
        scheme_ii_combine_partial_crt_fixed_kernel<8, 1><<<gridC, block2d, 0, stream>>>(
            C_partial, C, eA, eB, m, n, m_pad, n_pad, S,
            group_moduli, inv_table, M, M_half, P);
    } else {
        scheme_ii_combine_partial_crt_kernel<<<gridC, block2d, 0, stream>>>(
            C_partial, C, eA, eB, m, n, m_pad, n_pad, num_groups, S,
            group_moduli, inv_table, M, M_half, P);
    }
}

} // namespace ozaki
