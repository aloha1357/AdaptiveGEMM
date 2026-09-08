// include/int8_gemm.h
// PR1 contract: isolated row-major s8 x s8 -> s32 GEMM for sm_89.
// Only <128,128,64,3> is instantiated in PR1.
// Alignment contract: m%BM==0, n%BN==0, k%BK==0; else cudaErrorInvalidValue.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ozaki {

// ---------------------------------------------------------------------------
// Core INT8 GEMM launch
// ---------------------------------------------------------------------------

// v1 alignment contract: m % BM == 0, n % BN == 0, k % BK == 0.
// (BK, not MMA-k=32: the K loop steps by BK.)
// PR1 instantiates exactly one kernel: <128,128,64,3>.
// Returns cudaErrorInvalidValue if alignment contract is violated.
template<int BM, int BN, int BK, int Stages>
cudaError_t int8_gemm_s8s8s32_launch(const int8_t* A, const int8_t* B, int32_t* C,
                                      int m, int n, int k, cudaStream_t stream);

// Non-template convenience entry point (calls <128,128,64,3> specialization).
// Declared here, defined in int8_gemm.cu ??avoids specialization ordering issue.
cudaError_t int8_gemm_s8s8s32(const int8_t* A, const int8_t* B, int32_t* C,
                               int m, int n, int k, cudaStream_t stream);

// ---------------------------------------------------------------------------
// B.1 epilogue (PR4a). Not used by the cuBLAS backend.
// row_sum[i] = sum_k S_a[i,k]; col_sum[j] = sum_k S_b[k,j]  (precomputed)
// bias_a = (a==0) ? 0 : 128; bias_b = (b==0) ? 0 : 128
// Compensation is applied ONCE after the full-K MMA, not per BK tile:
//   acc_u = acc_mma + bias_b*row_sum[i] + bias_a*col_sum[j] + bias_a*bias_b*k
//   C[i,j] += ldexp((double)acc_u, eA[i] + eB[j] + slice_shift)
// ---------------------------------------------------------------------------
void int8_gemm_s8s8s32_f64_epilogue(
    const int8_t* A, const int8_t* B, double* C,
    const int32_t* eA, const int32_t* eB,
    const int32_t* row_sum, const int32_t* col_sum,
    int bias_a, int bias_b, int m, int n, int k, int slice_shift,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Benchmark vs cublasGemmEx INT8
// ---------------------------------------------------------------------------
struct Int8GemmBench {
    double our_ms;       // our kernel median over warmup+runs
    double cublas_ms;    // cublasGemmEx median
    double our_tops;     // 2*m*n*k / (our_ms*1e-3) / 1e12
    double cublas_tops;
    double peak_frac;    // our_tops / 121
    double vs_cublas;    // our_tops / cublas_tops
    int32_t max_abs_int32;
};

// Runs both kernels on a random s8 matrix of size m x n x k, returns bench.
// Uses the same row-major trick as the cuBLAS reference:
//   C = A*B (row-major) == cublasGemmEx(N,N, n,n,n, B,A,C, R_8I, COMPUTE_32I)
Int8GemmBench bench_int8_gemm_vs_cublas(int m, int n, int k);


void int8_gemm_run_crt_phase22_fused(
    const int8_t* A_mods, const int8_t* B_mods, int32_t* C_mods, double* C_out,
    int m, int n, int k, int m_pad, int n_pad, int k_pad,
    const int* primes, int num_moduli,
    double M_half, double scale_inv_sq,
    cudaStream_t stream);

} // namespace ozaki
