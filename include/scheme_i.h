#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <functional>
#include <cstdint>

#include "AdaptiveOzaki.h"

namespace ozaki {

inline int scheme_i_for_each_pair(int p, bool triangular, int T,
                                  const std::function<void(int a, int b)>& fn) {
    int n = 0;
    for (int a = 0; a < p; ++a)
      for (int b = 0; b < p; ++b)
        if (!triangular || a + b <= T) { fn(a, b); ++n; }
    return n;
}

void scheme_i_pack(const double* d_A, const double* d_B, double* d_C, 
                   int m, int n, int k, int mantissa_bits, int p, int bits_per_int8,
                   int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                   int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum,
                   bool enable_skip_scalA, bool& scalA_cached, bool fused_scale_split,
                   bool enable_stage_profiling, std::vector<PipelineStage>& stages, cudaStream_t stream = 0);

void execute_scheme_i_v0(cublasHandle_t h, const double* d_A, const double* d_B, double* d_C, 
                         int m, int n, int k, int mantissa_bits, int p, int bits_per_int8, bool triangular, int T,
                         int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                         int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum,
                         int32_t* C_i32, bool enable_stage_profiling, std::vector<PipelineStage>& stages);

void execute_scheme_i_b1(const double* d_A, const double* d_B, double* d_C, 
                         int m, int n, int k, int mantissa_bits, int p, int bits_per_int8, bool triangular, int T,
                         int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                         int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum, int32_t* C_i32);

void execute_scheme_i_fused(const double* d_A, const double* d_B, double* d_C, 
                            int m, int n, int k, int mantissa_bits, int p, int bits_per_int8, bool triangular, int T,
                            int8_t* A_slices, int8_t* B_slices, double* maxA, double* maxB,
                            int32_t* eA, int32_t* eB, int32_t* row_sum, int32_t* col_sum, int32_t* C_i32);

// test helpers
void launch_scheme_i_reduce_maxB(const double* B, double* maxB, int k, int n);
void launch_scheme_i_exponents_from_max(const double* maxAbs, int32_t* e, int n);
void launch_test_pair_table(int8_t qA_val, int8_t qB_val, double* d_C);

void launch_fused_scale_split_A(const double* A, int8_t* slices, int32_t* eA, int32_t* row_sum,
                                int m, int k, int p, int mantissa_bits, int bits_per_int8, cudaStream_t stream = 0);

}
