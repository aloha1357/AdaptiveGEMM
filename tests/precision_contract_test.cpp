#include "AdaptiveOzaki.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace ozaki;

int main() {
    OzakiConfig config;
    config.mode = ExecutionMode::SchemeI;
    config.scheme_i_num_split = 3;
    config.scheme_i_bits_per_int8 = 8;
    config.target_mantissa_bits = 53;
    
    AdaptiveOzakiEngine engine(config);
    
    double h_A[1] = {1.0};
    double h_B[1] = {1.0};
    double h_C[1] = {0.0};
    
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_B, sizeof(double));
    cudaMalloc(&d_C, sizeof(double));
    cudaMemcpy(d_A, h_A, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(double), cudaMemcpyHostToDevice);
    
    GEMMInput input;
    input.M = 1; input.N = 1; input.K = 1;
    
    engine.execute(d_A, d_B, d_C, input);
    
    auto contract = engine.getPrecisionContract();
    
    if (contract.mode != "SchemeI") {
        std::cerr << "Contract mode wrong: " << contract.mode << "\n";
        return 1;
    }
    if (contract.num_split != 3) {
        std::cerr << "Contract num_split wrong: " << contract.num_split << "\n";
        return 1;
    }
    
    std::cout << "Precision contract tests passed!\n";
    return 0;
}
