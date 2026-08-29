#pragma once

#include <cublas_v2.h>

namespace ozaki {

void execute_scheme_ii_gpu(cublasHandle_t handle, const double* d_A, const double* d_B, double* d_C, 
                           int m, int n, int k, int num_moduli);

void execute_phase22_fused(cublasHandle_t handle, const double* d_A, const double* d_B, double* d_C, 
                           int m, int n, int k, int num_moduli);

} // namespace ozaki
