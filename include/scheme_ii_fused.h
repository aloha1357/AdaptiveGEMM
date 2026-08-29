#pragma once

#include <cublas_v2.h>
#include "AdaptiveOzaki.h"

namespace ozaki {

// Staged / semi-fused SchemeII executor (M7.1-M7.4).
// Not a true register-level fused CRT kernel (M7.5).
// Pipeline: exp A/B -> quantize residues -> per-modulus INT8 GEMM -> CRT + undo scale.
// Optional CUDA Graph capture for fixed-shape repeat workloads.
void execute_scheme_ii_fused(cublasHandle_t handle, const double* d_A, const double* d_B, double* d_C,
                             int m, int n, int k, const OzakiConfig& config, AdaptiveOzakiEngine* engine);

} // namespace ozaki
