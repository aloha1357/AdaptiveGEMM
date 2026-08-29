#include "int8_gemm.h"
#include <cuComplex.h>
#include "scheme_i.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        std::exit(1); \
    } \
}

namespace ozaki {

__global__ void scheme_i_reduce_maxA(const double* A, double* maxA, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double max_val = 0.0;
        for (int j = 0; j < k; ++j) {
            double v = fabs(A[(size_t)i * k + j]);
            if (v > max_val) max_val = v;
        }
        maxA[i] = max_val;
    }
}

__global__ void scheme_i_reduce_maxB(const double* B, double* maxB, int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < k) {
        double v = fabs(B[(size_t)row * n + col]);
        unsigned long long int* addr = (unsigned long long int*)&maxB[col];
        atomicMax(addr, __double_as_longlong(v));
    }
}

__global__ void scheme_i_exponents_from_max(const double* maxAbs, int32_t* e, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double v = maxAbs[i];
        e[i] = (v == 0.0) ? 0 : (int)ceil(log2(v));
    }
}

__global__ void scheme_i_write_slices_A(const double* A, const int32_t* eA,
    int8_t* slices, int32_t* row_sum, int m, int k, int p, int mantissa_bits, int bits_per_int8) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        int e = eA[row];
        int32_t sum[10]; // up to max p
        for(int a=0; a<p; a++) sum[a] = 0;
        
        for (int col = 0; col < k; ++col) {
            double x = A[(size_t)row * k + col];
            double scaled = ldexp(x, -e);
            double q_double = round(scaled * ldexp(1.0, mantissa_bits));
            long long q = (long long)q_double;
            
            for (int a = 0; a < p; ++a) {
                int8_t S;
                if (a == 0) {
                    S = (int8_t)(q >> (bits_per_int8 * (p - 1)));
                } else {
                    int mask = (1 << bits_per_int8) - 1;
                    int bias = 1 << (bits_per_int8 - 1);
                    int u = (q >> (bits_per_int8 * (p - 1 - a))) & mask;
                    S = (int8_t)(u - bias);
                }
                slices[(size_t)a * m * k + (size_t)row * k + col] = S;
                sum[a] += S;
            }
        }
        for(int a=0; a<p; a++) row_sum[a * m + row] = sum[a];
    }
}

__global__ void scheme_i_write_slices_B(const double* B, const int32_t* eB,
    int8_t* slices, int32_t* col_sum, int k, int n, int p, int mantissa_bits, int bits_per_int8) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        int e = eB[col];
        int32_t sum[10];
        for(int b=0; b<p; b++) sum[b] = 0;
        
        for (int row = 0; row < k; ++row) {
            double x = B[(size_t)row * n + col];
            double scaled = ldexp(x, -e);
            double q_double = round(scaled * ldexp(1.0, mantissa_bits));
            long long q = (long long)q_double;
            
            for (int b = 0; b < p; ++b) {
                int8_t S;
                if (b == 0) {
                    S = (int8_t)(q >> (bits_per_int8 * (p - 1)));
                } else {
                    int mask = (1 << bits_per_int8) - 1;
                    int bias = 1 << (bits_per_int8 - 1);
                    int u = (q >> (bits_per_int8 * (p - 1 - b))) & mask;
                    S = (int8_t)(u - bias);
                }
                slices[(size_t)b * k * n + (size_t)row * n + col] = S;
                sum[b] += S;
            }
        }
        for(int b=0; b<p; b++) col_sum[b * n + col] = sum[b];
    }
}

__global__ void accumulate_i32_to_f64(double* C, const int32_t* C_i32,
    const int32_t* eA, const int32_t* eB,
    const int32_t* row_sum, const int32_t* col_sum,
    int bias_a, int bias_b, int m, int n, int k, int slice_shift) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        int64_t acc_mma = C_i32[(size_t)row * n + col]; 
        int64_t bias_b_64 = bias_b;
        int64_t bias_a_64 = bias_a;
        int64_t rs = row_sum[row];
        int64_t cs = col_sum[col];
        int64_t acc_u = acc_mma 
                      + bias_b_64 * rs 
                      + bias_a_64 * cs 
                      + bias_a_64 * bias_b_64 * k;
        
        double val = ldexp((double)acc_u, eA[row] + eB[col] + slice_shift);
        C[(size_t)row * n + col] += val;
    }
}

__global__ void naive_int8_gemm(int m, int n, int k, const int8_t* A, const int8_t* B, int32_t* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        int32_t sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += (int32_t)A[row * k + i] * (int32_t)B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void scheme_i_pack(const double* d_A, const double* d_B, double* d_C, 
                   int m, int n, int k, int mantissa_bits, int p, int bits_per_int8,
                   int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                   int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum,
                   bool enable_skip_scalA, bool& scalA_cached, bool fused_scale_split,
                   bool enable_stage_profiling, std::vector<PipelineStage>& stages, cudaStream_t stream) {
    cudaMemsetAsync(d_C, 0, (size_t)m * n * sizeof(double), stream);
    cudaMemsetAsync(maxB, 0, n * sizeof(double), stream);
    
    cudaEvent_t t0, t1;
    if (enable_stage_profiling) {
        cudaEventCreate(&t0); cudaEventCreate(&t1);
    }
    
    int threads = 256;
    dim3 gridB((n + 31) / 32, (k + 31) / 32);
    dim3 blockB(32, 32);
    
    if (!enable_skip_scalA || !scalA_cached) {
        if (enable_stage_profiling) cudaEventRecord(t0, stream);
        
        launch_fused_scale_split_A(d_A, A_slices, eA, row_sum, m, k, p, mantissa_bits, bits_per_int8, stream);
        
        if (enable_stage_profiling) {
            cudaEventRecord(t1, stream);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            PipelineStage stage;
            stage.stage_name = "scale_A";
            stage.stage_timing = ms;
            stages.push_back(stage);
            
            PipelineStage split_stage;
            split_stage.stage_name = "split";
            split_stage.stage_timing = 0.001f; // fused
            stages.push_back(split_stage);
        }
        scalA_cached = true;
    } else {
        if (enable_stage_profiling) {
            PipelineStage stage;
            stage.stage_name = "scale_A";
            stage.stage_timing = 0.0f;
            stages.push_back(stage);
            
            PipelineStage split_stage;
            split_stage.stage_name = "split";
            split_stage.stage_timing = 0.0f;
            stages.push_back(split_stage);
        }
    }
    
    if (enable_stage_profiling) cudaEventRecord(t0, stream);
    
    scheme_i_reduce_maxB<<<gridB, blockB, 0, stream>>>(d_B, maxB, k, n);
    CHECK_CUDA(cudaGetLastError());
    scheme_i_exponents_from_max<<<(n + threads - 1) / threads, threads, 0, stream>>>(maxB, eB, n);
    CHECK_CUDA(cudaGetLastError());
    scheme_i_write_slices_B<<<(n + threads - 1) / threads, threads, 0, stream>>>(d_B, eB, B_slices, col_sum, k, n, p, mantissa_bits, bits_per_int8);
    CHECK_CUDA(cudaGetLastError());
    
    if (enable_stage_profiling) {
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        PipelineStage stage;
        stage.stage_name = "scale_B";
        stage.stage_timing = ms;
        stages.push_back(stage);
        
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
}

void execute_scheme_i_v0(cublasHandle_t h, const double* d_A, const double* d_B, double* d_C, 
                         int m, int n, int k, int mantissa_bits, int p, int bits_per_int8, bool triangular, int T,
                         int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                         int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum,
                         int32_t* C_i32, bool enable_stage_profiling, std::vector<PipelineStage>& stages) {
    cudaEvent_t t0, t1;
    if (enable_stage_profiling) {
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
    }
    
    std::vector<int> pairs_a, pairs_b;
    scheme_i_for_each_pair(p, triangular, T, [&](int a, int b) {
        pairs_a.push_back(a);
        pairs_b.push_back(b);
    });
    
    int num_pairs = pairs_a.size();
    
    std::vector<const void*> A_ptrs(num_pairs);
    std::vector<const void*> B_ptrs(num_pairs);
    std::vector<void*> C_ptrs(num_pairs);
    
    int32_t* d_C_i32_batched;
    cudaMalloc(&d_C_i32_batched, (size_t)num_pairs * m * n * sizeof(int32_t));
    cudaMemset(d_C_i32_batched, 0, (size_t)num_pairs * m * n * sizeof(int32_t));
    
    for (int i = 0; i < num_pairs; ++i) {
        int a = pairs_a[i];
        int b = pairs_b[i];
        A_ptrs[i] = A_slices + (size_t)a * m * k;
        B_ptrs[i] = B_slices + (size_t)b * k * n;
        C_ptrs[i] = d_C_i32_batched + (size_t)i * m * n;
    }
    
    const void** d_A_ptrs;
    const void** d_B_ptrs;
    void** d_C_ptrs;
    cudaMalloc(&d_A_ptrs, num_pairs * sizeof(void*));
    cudaMalloc(&d_B_ptrs, num_pairs * sizeof(void*));
    cudaMalloc(&d_C_ptrs, num_pairs * sizeof(void*));
    
    cudaMemcpy(d_A_ptrs, A_ptrs.data(), num_pairs * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ptrs, B_ptrs.data(), num_pairs * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ptrs, C_ptrs.data(), num_pairs * sizeof(void*), cudaMemcpyHostToDevice);
    
    int32_t alpha = 1, beta = 0;
    
    // According to cublas documentation, cublasGemmBatchedEx takes arrays of pointers.
    // However, A is transposed/non-transposed?
    // We are doing C = A * B. B_slices is the second matrix, A_slices is the first.
    // wait, originally it was:
    // cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_slices, CUDA_R_8I, n, A_slices, CUDA_R_8I, k, &beta, C_i32, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cublasStatus_t status = cublasGemmBatchedEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha,
        d_B_ptrs, CUDA_R_8I, n,
        d_A_ptrs, CUDA_R_8I, k,
        &beta,
        d_C_ptrs, CUDA_R_32I, n,
        num_pairs,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS_ERROR " << status << " during batched gemm" << std::endl;
        std::exit(1);
    }
    
    if (enable_stage_profiling) {
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        PipelineStage stage;
        stage.stage_name = "gemm";
        stage.stage_timing = ms;
        stages.push_back(stage);
        
        cudaEventRecord(t0);
    }
    
    dim3 block(32, 8);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    
    for (int i = 0; i < num_pairs; ++i) {
        int a = pairs_a[i];
        int b = pairs_b[i];
        int sh = bits_per_int8 * ((p - 1 - a) + (p - 1 - b)) - 2 * mantissa_bits;
        int bias = 1 << (bits_per_int8 - 1);
        accumulate_i32_to_f64<<<grid, block>>>(d_C, d_C_i32_batched + (size_t)i * m * n, eA, eB,
                                       row_sum + a * m, col_sum + b * n,
                                       a == 0 ? 0 : bias, b == 0 ? 0 : bias, m, n, k, sh);
    }
    
    if (enable_stage_profiling) {
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        PipelineStage stage;
        stage.stage_name = "accumulate";
        stage.stage_timing = ms;
        stages.push_back(stage);
        
        PipelineStage undo_stage;
        undo_stage.stage_name = "undo_scale";
        undo_stage.stage_timing = 0.001f; // fused in accumulate
        stages.push_back(undo_stage);
        
        PipelineStage report_stage;
        report_stage.stage_name = "report";
        report_stage.stage_timing = 0.001f;
        stages.push_back(report_stage);
        
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    
    cudaFree(d_C_i32_batched);
    cudaFree(d_A_ptrs);
    cudaFree(d_B_ptrs);
    cudaFree(d_C_ptrs);
}

void execute_scheme_i_b1(const double* d_A, const double* d_B, double* d_C, 
                         int m, int n, int k, int mantissa_bits, int p, int bits_per_int8, bool triangular, int T,
                         int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                         int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum, int32_t* C_i32) {
    int launched = scheme_i_for_each_pair(p, triangular, T, [&](int a, int b) {
        int sh = bits_per_int8 * ((p - 1 - a) + (p - 1 - b)) - 2 * mantissa_bits;
        int bias = 1 << (bits_per_int8 - 1);
        if (k % 64 != 0) {
            dim3 block(16, 16);
            dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
            naive_int8_gemm<<<grid, block>>>(m, n, k, A_slices + (size_t)a * m * k, B_slices + (size_t)b * k * n, C_i32);
            dim3 b2(32, 8);
            dim3 g2((n + b2.x - 1) / b2.x, (m + b2.y - 1) / b2.y);
            accumulate_i32_to_f64<<<g2, b2>>>(d_C, C_i32, eA, eB,
                                           row_sum + a * m, col_sum + b * n,
                                           a == 0 ? 0 : bias, b == 0 ? 0 : bias, m, n, k, sh);
        } else {
            int8_gemm_s8s8s32_f64_epilogue(
                A_slices + (size_t)a * m * k, B_slices + (size_t)b * k * n, d_C, eA, eB,
                row_sum + a * m, col_sum + b * n,
                a == 0 ? 0 : bias, b == 0 ? 0 : bias, m, n, k, sh, 0);
        }
    });
}



void launch_scheme_i_reduce_maxB(const double* B, double* maxB, int k, int n) {
    dim3 gridB((n + 31) / 32, (k + 31) / 32);
    dim3 blockB(32, 32);
    scheme_i_reduce_maxB<<<gridB, blockB>>>(B, maxB, k, n);
}

void launch_scheme_i_exponents_from_max(const double* maxAbs, int32_t* e, int n) {
    int threads = 256;
    scheme_i_exponents_from_max<<<(n + threads - 1) / threads, threads>>>(maxAbs, e, n);
}

void launch_test_pair_table(int8_t qA_val, int8_t qB_val, double* d_C) {
    // A test-only helper to run p=2 pair table test
    // qA, qB can be -1 or 1.
    // For p=2, slices:
    int8_t hA[2], hB[2];
    int16_t qA = qA_val;
    hA[0] = (int8_t)(qA >> 8);
    hA[1] = (int8_t)((qA & 0xFF) - 128);
    int16_t qB = qB_val;
    hB[0] = (int8_t)(qB >> 8);
    hB[1] = (int8_t)((qB & 0xFF) - 128);
    
    int8_t *dA, *dB;
    cudaMalloc(&dA, 2); cudaMalloc(&dB, 2);
    cudaMemcpy(dA, hA, 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 2, cudaMemcpyHostToDevice);
    
    int32_t *dC32, *d_eA, *d_eB, *d_rs, *d_cs;
    cudaMalloc(&dC32, sizeof(int32_t));
    cudaMalloc(&d_eA, sizeof(int32_t)); cudaMalloc(&d_eB, sizeof(int32_t));
    cudaMalloc(&d_rs, 2 * sizeof(int32_t)); cudaMalloc(&d_cs, 2 * sizeof(int32_t));
    
    cudaMemset(dC32, 0, sizeof(int32_t));
    cudaMemset(d_eA, 0, sizeof(int32_t));
    cudaMemset(d_eB, 0, sizeof(int32_t));
    cudaMemset(d_C, 0, sizeof(double));
    
    int32_t rs[2] = {hA[0], hA[1]};
    int32_t cs[2] = {hB[0], hB[1]};
    cudaMemcpy(d_rs, rs, 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cs, cs, 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    int p = 2;
    int mantissa_bits = 8;
    for (int a = 0; a < p; ++a) {
        for (int b = 0; b < p; ++b) {
            dim3 grid(1,1), block(1,1);
            naive_int8_gemm<<<grid, block>>>(1, 1, 1, dA + a, dB + b, dC32);
            int sh = 8 * ((p - 1 - a) + (p - 1 - b)) - 2 * mantissa_bits;
            accumulate_i32_to_f64<<<grid, block>>>(d_C, dC32, d_eA, d_eB,
                                                   d_rs + a, d_cs + b,
                                                   a == 0 ? 0 : 128, b == 0 ? 0 : 128, 1, 1, 1, sh);
        }
    }
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC32);
    cudaFree(d_eA); cudaFree(d_eB); cudaFree(d_rs); cudaFree(d_cs);
}

} // namespace ozaki
