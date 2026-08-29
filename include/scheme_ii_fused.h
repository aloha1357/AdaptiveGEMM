#pragma once

#include <cublas_v2.h>
#include "AdaptiveOzaki.h"

namespace ozaki {

// SchemeII executor (M7 staged, M8 custom MMA).
// M8.1: per-modulus custom MMA, still writes C_mod.
// M8.2: grouped custom MMA (group_size 2/4), still writes C_mod.
// M8.3: grouped partial CRT epilogue, writes C mod M_g then a combine kernel.
// M8.4: true fused CRT epilogue writes double C when group_size >= num_moduli.
void execute_scheme_ii_fused(cublasHandle_t handle, const double* d_A, const double* d_B, double* d_C,
                             int m, int n, int k, const OzakiConfig& config, AdaptiveOzakiEngine* engine);

} // namespace ozaki
