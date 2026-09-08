#include "AdaptiveOzaki.h"
#include "DoubleDouble.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace ozaki;

int main(int argc, char** argv) {
    OzakiConfig config;
    config.mode = ExecutionMode::SchemeI;
    config.scheme_i_num_split = 3;
    config.scheme_i_bits_per_int8 = 8;
    config.target_mantissa_bits = 53;
    
    bool use_dd = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input-type" && i + 1 < argc) {
            std::string t = argv[++i];
            if (t == "DoubleDouble") {
                use_dd = true;
            }
        }
    }
    
    AdaptiveOzakiEngine engine(config);
    
    if (use_dd) {
        DoubleDouble h_A[1] = {DoubleDouble(1.0, 0.0)};
        DoubleDouble h_B[1] = {DoubleDouble(1.0, 0.0)};
        DoubleDouble h_C[1] = {DoubleDouble(0.0, 0.0)};
        DoubleDouble *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sizeof(DoubleDouble));
        cudaMalloc(&d_B, sizeof(DoubleDouble));
        cudaMalloc(&d_C, sizeof(DoubleDouble));
        cudaMemcpy(d_A, h_A, sizeof(DoubleDouble), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeof(DoubleDouble), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, sizeof(DoubleDouble), cudaMemcpyHostToDevice);
        
        GEMMInput input;
        input.M = 1; input.N = 1; input.K = 1;
        engine.execute(d_A, d_B, d_C, input);
        
        auto contract = engine.getPrecisionContract();
        std::cout << "Contract mode: " << contract.mode << "\n";
        std::cout << "Contract input_type: " << contract.input_type << "\n";
        std::cout << "Contract crt_bits: " << contract.crt_bits << "\n";
        std::cout << "Contract required_bits: " << contract.required_crt_bits << "\n";
        
        if (contract.input_type != "DoubleDouble") {
            std::cerr << "Contract input_type wrong: " << contract.input_type << "\n";
            return 1;
        }
        std::cout << "Precision contract tests passed!\n";
        return 0;
    }
    
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
    std::cout << "Contract mode: " << contract.mode << "\n";
    std::cout << "Contract input_type: " << contract.input_type << "\n";
    std::cout << "Contract num_split: " << contract.num_split << "\n";
    
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
