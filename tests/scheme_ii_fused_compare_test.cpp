#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "AdaptiveOzaki.h"

static void generate_random_data(std::vector<double>& data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
}

int main(int argc, char** argv) {
    int m = 512, n = 512, k = 512;
    int num_moduli = 0;
    std::string policy_str = "Auto";
    bool use_cuda_graph = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-M" || arg == "--M") && i + 1 < argc) m = std::atoi(argv[++i]);
        else if ((arg == "-N" || arg == "--N") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if ((arg == "-K" || arg == "--K") && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (arg == "--num_moduli" && i + 1 < argc) num_moduli = std::atoi(argv[++i]);
        else if (arg == "--precision_policy" && i + 1 < argc) policy_str = argv[++i];
        else if (arg == "--use_cuda_graph") use_cuda_graph = true;
    }

    std::vector<double> A((size_t)m * k);
    std::vector<double> B((size_t)k * n);
    std::vector<double> C_fused((size_t)m * n, 0.0);
    std::vector<double> C_scheme_ii((size_t)m * n, 0.0);
    std::vector<double> C_ref((size_t)m * n, 0.0);

    generate_random_data(A);
    generate_random_data(B);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size_t)m * k * sizeof(double));
    cudaMalloc(&d_B, (size_t)k * n * sizeof(double));
    cudaMalloc(&d_C, (size_t)m * n * sizeof(double));

    cudaMemcpy(d_A, A.data(), (size_t)m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), (size_t)k * n * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    auto make_config = [&](ozaki::ExecutionMode mode) {
        ozaki::OzakiConfig config;
        config.mode = mode;
        config.scheme_ii_num_moduli = num_moduli;
        config.target_mantissa_bits = 53;
        config.scheme_ii_use_cuda_graph = (mode == ozaki::ExecutionMode::SchemeII_Fused) && use_cuda_graph;
        if (policy_str == "Auto" || num_moduli <= 0) {
            config.precision_policy = ozaki::PrecisionPolicy::Auto;
        } else {
            config.precision_policy = ozaki::PrecisionPolicy::Fixed;
        }
        return config;
    };

    ozaki::OzakiConfig config_unfused = make_config(ozaki::ExecutionMode::SchemeII);
    ozaki::AdaptiveOzakiEngine engine_unfused(config_unfused);
    engine_unfused.execute(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C_scheme_ii.data(), d_C, (size_t)m * n * sizeof(double), cudaMemcpyDeviceToHost);
    engine_unfused.freeWorkspace();

    ozaki::OzakiConfig config_fused = make_config(ozaki::ExecutionMode::SchemeII_Fused);
    ozaki::AdaptiveOzakiEngine engine_fused(config_fused);
    engine_fused.allocateWorkspace(m, n, k);
    engine_fused.execute(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C_fused.data(), d_C, (size_t)m * n * sizeof(double), cudaMemcpyDeviceToHost);

    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    cudaMemcpy(C_ref.data(), d_C, (size_t)m * n * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "C_fused[0] = " << C_fused[0] << std::endl;
    std::cout << "C_scheme_ii[0] = " << C_scheme_ii[0] << std::endl;
    std::cout << "C_ref[0] = " << C_ref[0] << std::endl;

    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double max_abs_vs_ref = 0.0;
    for (int i = 0; i < m * n; ++i) {
        double abs_diff = std::abs(C_fused[i] - C_scheme_ii[i]);
        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;

        double abs_ref = std::abs(C_fused[i] - C_ref[i]);
        if (abs_ref > max_abs_vs_ref) max_abs_vs_ref = abs_ref;

        double rel_diff = abs_ref / std::max(1e-10, std::abs(C_ref[i]));
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
    }

    auto contract = engine_fused.getPrecisionContract();
    bool passed_precision_gate = (max_rel_diff <= 1e-6);
    bool passed = (max_abs_diff <= 1e-10) && passed_precision_gate &&
                  !contract.native_fp64_used && !contract.residual_enabled;

    std::cout << "Max absolute difference (Fused vs Unfused): " << max_abs_diff << std::endl;
    std::cout << "Max relative difference (Fused vs cuBLAS): " << max_rel_diff << std::endl;
    std::cout << "max_abs_error=" << max_abs_vs_ref << std::endl;
    std::cout << "max_relative_error=" << max_rel_diff << std::endl;
    std::cout << "fallback_mode=" << (contract.fallback_mode.empty() ? "None" : contract.fallback_mode) << std::endl;
    std::cout << "num_moduli=" << contract.num_moduli << std::endl;
    std::cout << "passed_precision_gate=" << (passed_precision_gate ? 1 : 0) << std::endl;
    std::cout << "used_cuda_graph=" << (contract.used_cuda_graph ? "true" : "false") << std::endl;
    std::cout << "used_true_fused_kernel=" << (contract.used_true_fused_kernel ? "true" : "false") << std::endl;
    std::cout << "used_scheme_ii_fused=" << (contract.used_scheme_ii_fused ? "true" : "false") << std::endl;
    std::cout << "native_fp64_used=" << (contract.native_fp64_used ? "true" : "false") << std::endl;
    std::cout << "residual_enabled=" << (contract.residual_enabled ? "true" : "false") << std::endl;

    if (passed) {
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED" << std::endl;
    }

    engine_fused.freeWorkspace();
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return passed ? 0 : 1;
}
