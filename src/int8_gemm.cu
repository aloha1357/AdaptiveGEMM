#include "int8_gemm.h"
#include "mma_sm89.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

template <int BM, int BN, int BK, int Stages>
__global__ __launch_bounds__(256, 2)
void int8_gemm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B, int32_t* __restrict__ C, int m, int n, int k) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int wm = wid / 2;
    int wn = wid % 2;

    int bm_off = blockIdx.y * BM;
    int bn_off = blockIdx.x * BN;

    extern __shared__ int4 smem_int4[];
    int8_t* smem = (int8_t*)smem_int4;
    int8_t* sA_ptr = smem;
    int8_t* sB_ptr = smem + Stages * 128 * 128;

    auto swizzle_A = [](int row, int col) {
        return col ^ ((row & 7) << 4);
    };
    
    // sB is [128][64] (col-major in N). So row dimension is N (128 rows), col dimension is K (64 columns).
    // The row length is 64 bytes. We must use a 64-byte bank conflict swizzle that preserves 16-byte alignment.
    auto swizzle_B = [](int row, int col) {
        return col ^ ((row & 3) << 4);
    };

    uint32_t a_frag[2][2][4];
    uint32_t b_frag[2][8][2];
    int32_t acc[2][8][4] = {};

    auto load_a_smem = [&](int buf, int k_offset) {
        int8_t* sA = sA_ptr + buf * 128 * 128;
        for (int i = tid; i < BM * BK; i += 256) {
            int row = i / BK;
            int col = i % BK;
            int gm = bm_off + row;
            int gk = k_offset + col;
            int8_t val = (gm < m && gk < k) ? A[(size_t)gm * k + gk] : 0;
            sA[row * 128 + swizzle_A(row, col)] = val;
        }
    };

    auto load_b_smem = [&](int buf, int k_offset) {
        int8_t* sB = sB_ptr + buf * 128 * 64;
        for (int i = tid; i < BK * BN; i += 256) {
            int row = i / BK; // N! (0 to 127)
            int col = i % BK; // K! (0 to 63)
            int gn = bn_off + row;
            int gk = k_offset + col;
            int8_t val = (gk < k && gn < n) ? B[(size_t)gk * n + gn] : 0;
            sB[row * 64 + swizzle_B(row, col)] = val;
        }
    };

    auto load_a_frag = [&](int buf, int k_step, int fbuf) {
        int8_t* sA = sA_ptr + buf * 128 * 128;
        int ms = wm * 32;
        for (int i = 0; i < 2; i++) {
            int a_row = ms + i * 16 + (lane % 16);
            int a_col = k_step + (lane / 16) * 16;
            uint32_t addr = __cvta_generic_to_shared(&sA[a_row * 128 + swizzle_A(a_row, a_col)]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(a_frag[fbuf][i][0]), "=r"(a_frag[fbuf][i][1]), "=r"(a_frag[fbuf][i][2]), "=r"(a_frag[fbuf][i][3])
                : "r"(addr));
        }
    };

    auto load_b_frag = [&](int buf, int k_step, int fbuf) {
        int8_t* sB = sB_ptr + buf * 128 * 64;
        int ns = wn * 64;
        for (int j = 0; j < 8; j++) {
            int b_row = ns + j * 8 + (lane % 8); // N!
            int b_col = k_step + (lane / 8 % 2) * 16; // K! (0, 16, 32, 48)
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
            prefetch_k += BK;
        }
    }
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
            prefetch_k += BK;
        }

        if (next_k1 == 0 && do_prefetch) {
            __syncthreads();
        }

        if (do_prefetch) {
            load_a_frag(next_stage, next_k1, p ^ 1);
            load_b_frag(next_stage, next_k1, p ^ 1);
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 8; j++) {
                mma_m16n8k32_s8(acc[i][j], a_frag[p][i], b_frag[p][j], acc[i][j]);
            }
        }
        
        k1 = next_k1;
        read_stage = next_stage;
    }

    const int r_off[4] = { lane/4, lane/4, lane/4+8, lane/4+8 };
    const int c_off[4] = { (lane%4)*2, (lane%4)*2+1, (lane%4)*2, (lane%4)*2+1 };

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int r = 0; r < 4; ++r) {
                int gm = bm_off + wm * 32 + i*16 + r_off[r];
                int gn = bn_off + wn * 64 + j*8  + c_off[r];
                if (gm < m && gn < n) {
                    C[(size_t)gm * n + gn] = acc[i][j][r];
                }
            }
        }
    }
}

namespace ozaki {
template<int BM, int BN, int BK, int Stages>
cudaError_t int8_gemm_s8s8s32_launch(const int8_t* A, const int8_t* B, int32_t* C,
                                      int m, int n, int k, cudaStream_t stream) {
    if (m % BM != 0 || n % BN != 0 || k % BK != 0) return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid(n / BN, m / BM);
    int smem_size = Stages * 128 * 128 + Stages * 128 * 64;
    cudaFuncSetAttribute(int8_gemm_kernel<BM,BN,BK,Stages>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    int8_gemm_kernel<BM,BN,BK,Stages><<<grid, block, smem_size, stream>>>(A, B, C, m, n, k);
    return cudaGetLastError();
}

cudaError_t int8_gemm_s8s8s32(const int8_t* A, const int8_t* B, int32_t* C,
                               int m, int n, int k, cudaStream_t stream) {
    return int8_gemm_s8s8s32_launch<128, 128, 64, 3>(A, B, C, m, n, k, stream);
}
}
