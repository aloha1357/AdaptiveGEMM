#pragma once

namespace ozaki {

void execute_scheme_ii(const double* d_A, const double* d_B, double* d_C, 
                       int m, int n, int k, int num_moduli);

} // namespace ozaki
