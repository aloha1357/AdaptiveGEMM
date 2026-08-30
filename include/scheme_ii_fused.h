#pragma once

#include <cublas_v2.h>
#include "AdaptiveOzaki.h"

namespace ozaki {

// SchemeII executor (M7 staged, M8 custom MMA, M9 speed-path autotune).
// M8.1: per-modulus custom MMA, still writes C_mod.
// M8.2: grouped custom MMA (group_size 2/4), still writes C_mod.
// M8.3 speed path: group_size=4 + --use_true_fused_kernel
//      used_partial_crt_epilogue=true, used_true_fused_kernel=false.
//      A/B residues are still materialized unless tile-local prologue is on.
// M8.4 control: group_size >= num_moduli + --use_true_fused_kernel
//      used_partial_crt_epilogue=false, used_true_fused_kernel=true.
void execute_scheme_ii_fused(cublasHandle_t handle, const double* d_A, const double* d_B, double* d_C,
                             int m, int n, int k, const OzakiConfig& config, AdaptiveOzakiEngine* engine);

} // namespace ozaki
