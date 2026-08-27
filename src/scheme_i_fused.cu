#include "scheme_i.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        std::exit(1); \
    } \
}

namespace ozaki {

void scheme_i_pack(const double* d_A, const double* d_B, double* d_C, 
                   int m, int n, int k, int mantissa_bits, int p, int bits_per_int8,
                   int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                   int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum);

__global__ void fused_scheme_i_kernel(
    const double* __restrict__ d_C_in, double* __restrict__ d_C,
    const int8_t* __restrict__ A_slices, const int8_t* __restrict__ B_slices,
    const int32_t* __restrict__ eA, const int32_t* __restrict__ eB,
    const int32_t* __restrict__ row_sum, const int32_t* __restrict__ col_sum,
    int m, int n, int k, int p, int mantissa_bits, int bits_per_int8, bool triangular, int T) 
{
    const int BM = 128;
    const int BN = 64;
    const int BK = 32;

    // p <= 5
    __shared__ int8_t sa[5][BM][BK];
    __shared__ int8_t sb[5][BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    // Each thread computes an 8x4 tile of C
    // 16x16 threads = 256 threads
    // 256 * 8x4 = 128x64 tile
    
    double acc_f64[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc_f64[i][j] = 0.0;
        }
    }

    int num_k_tiles = k / BK;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // Load A slices
        for (int i = tid; i < p * BM * BK; i += blockDim.x * blockDim.y) {
            int local_a = i / (BM * BK);
            int rem = i % (BM * BK);
            int local_r = rem / BK;
            int local_c = rem % BK;
            int global_r = blockIdx.y * BM + local_r;
            int global_c = k_tile * BK + local_c;
            sa[local_a][local_r][local_c] = A_slices[local_a * m * k + global_r * k + global_c];
        }

        // Load B slices
        for (int i = tid; i < p * BN * BK; i += blockDim.x * blockDim.y) {
            int local_b = i / (BN * BK);
            int rem = i % (BN * BK);
            int local_r = rem / BN;   
            int local_c = rem % BN;
            int global_r = k_tile * BK + local_r;
            int global_c = blockIdx.x * BN + local_c;
            sb[local_b][local_r][local_c] = B_slices[local_b * k * n + global_r * n + global_c];
        }

        __syncthreads();

        for (int a = 0; a < p; ++a) {
            for (int b = 0; b < p; ++b) {
                if (triangular && a + b > T) continue;
                
                int32_t i32_frag[8][4];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) i32_frag[i][j] = 0;
                }
                
                for (int kk = 0; kk < BK; ++kk) {
                    int8_t reg_a[8];
                    int8_t reg_b[4];
                    #pragma unroll
                    for (int i = 0; i < 8; i++) reg_a[i] = sa[a][ty * 8 + i][kk];
                    #pragma unroll
                    for (int j = 0; j < 4; j++) reg_b[j] = sb[b][kk][tx * 4 + j];
                    
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            i32_frag[i][j] += (int32_t)reg_a[i] * (int32_t)reg_b[j];
                        }
                    }
                }
                
                int sh = bits_per_int8 * ((p - 1 - a) + (p - 1 - b)) - 2 * mantissa_bits;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        acc_f64[i][j] += ldexp((double)i32_frag[i][j], sh);
                    }
                }
            }
        }
        __syncthreads();
    }

    int global_row_base = blockIdx.y * BM + ty * 8;
    int global_col_base = blockIdx.x * BN + tx * 4;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            int r = global_row_base + i;
            int c = global_col_base + j;
            
            if (r < m && c < n) {
                double bias_sum = 0.0;
                for (int a = 0; a < p; ++a) {
                    for (int b = 0; b < p; ++b) {
                        if (triangular && a + b > T) continue;
                        int bias_val = 1 << (bits_per_int8 - 1);
                        int bias_a_val = (a == 0) ? 0 : bias_val;
                        int bias_b_val = (b == 0) ? 0 : bias_val;
                        int32_t rs = row_sum[a * m + r];
                        int32_t cs = col_sum[b * n + c];
                        
                        int64_t bias_b_64 = bias_b_val;
                        int64_t bias_a_64 = bias_a_val;
                        int64_t term = bias_b_64 * rs + bias_a_64 * cs + bias_a_64 * bias_b_64 * k;
                        
                        int sh = bits_per_int8 * ((p - 1 - a) + (p - 1 - b)) - 2 * mantissa_bits;
                        bias_sum += ldexp((double)term, sh);
                    }
                }
                
                double val = acc_f64[i][j] + bias_sum;
                val = ldexp(val, eA[r] + eB[c]);
                d_C[r * n + c] += val;
            }
        }
    }
}

void execute_scheme_i_fused(const double* d_A, const double* d_B, double* d_C, 
                            int m, int n, int k, int mantissa_bits, int p, int bits_per_int8, bool triangular, int T,
                            int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                            int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum, int32_t* C_i32) 
{
    if (m % 128 != 0 || n % 64 != 0 || k % 32 != 0) {
        execute_scheme_i_b1(d_A, d_B, d_C, m, n, k, mantissa_bits, p, bits_per_int8, triangular, T, A_slices, B_slices, maxA, maxB, eA, eB, row_sum, col_sum, C_i32);
        return;
    }

    scheme_i_pack(d_A, d_B, d_C, m, n, k, mantissa_bits, p, bits_per_int8, A_slices, B_slices, maxA, maxB, eA, eB, row_sum, col_sum);

    const int BM = 128;
    const int BN = 64;
    
    dim3 block(16, 16);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    
    fused_scheme_i_kernel<<<grid, block>>>(d_C, d_C, A_slices, B_slices, eA, eB, row_sum, col_sum, m, n, k, p, mantissa_bits, bits_per_int8, triangular, T);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace ozaki
