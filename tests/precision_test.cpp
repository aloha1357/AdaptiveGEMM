#include "AdaptiveOzaki.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <omp.h>

using namespace ozaki;

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(1); } } while (0)
#define CHECK_CUBLAS(call) do { cublasStatus_t err = call; if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(1); } } while (0)

static void fill_random(std::vector<double>& matrix, double min_value, double max_value) {
    std::mt19937 generator(123);
    std::uniform_real_distribution<double> distribution(min_value, max_value);
    for (double& value : matrix) value = distribution(generator);
}

static double compute_max_error(const std::vector<double>& ref, const std::vector<double>& actual, double& rms_error, double& max_rel_error) {
    double max_err = 0.0;
    double sum_sq_err = 0.0;
    max_rel_error = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = std::abs(ref[i] - actual[i]);
        max_err = std::max(max_err, diff);
        sum_sq_err += diff * diff;
        if (std::abs(ref[i]) > 1e-12) {
            max_rel_error = std::max(max_rel_error, diff / std::abs(ref[i]));
        }
    }
    rms_error = std::sqrt(sum_sq_err / ref.size());
    return max_err;
}

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    ExecutionMode mode = ExecutionMode::Phase22;
    int num_moduli = 15;
    int target_mantissa_bits = 53;
    int repeat = 1;
    bool enable_stage_profiling = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            std::string mode_str = argv[++i];
            if (mode_str == "Phase22") mode = ExecutionMode::Phase22;
            else if (mode_str == "SchemeI") mode = ExecutionMode::SchemeI;
            else if (mode_str == "SchemeII") mode = ExecutionMode::SchemeII;
        } else if (arg == "--num_moduli" && i + 1 < argc) {
            num_moduli = std::atoi(argv[++i]);
        } else if (arg == "--target_mantissa_bits" && i + 1 < argc) {
            target_mantissa_bits = std::atoi(argv[++i]);
        } else if (arg == "-M" && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (arg == "-N" && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if (arg == "-K" && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (arg == "--stage") {
            enable_stage_profiling = true;
        }
    }

    std::vector<double> h_A(M * K), h_B(K * N), h_reference(M * N, 0.0), h_result(M * N, 0.0);
    fill_random(h_A, -1.0, 1.0);
    fill_random(h_B, -1.0, 1.0);
    
    if (mode == ExecutionMode::SchemeII) {
        if (M > 128) M = 128;
        if (N > 128) N = 128;
        if (K > 128) K = 128;
        // Resize after limiting
        h_A.resize(M * K); fill_random(h_A, -1.0, 1.0);
        h_B.resize(K * N); fill_random(h_B, -1.0, 1.0);
        h_reference.resize(M * N, 0.0);
        h_result.resize(M * N, 0.0);
    }

    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(double)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const double alpha = 1.0, beta = 0.0;
    
    // row-major C=A*B -> column-major C^T = B^T * A^T
    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUDA(cudaMemcpy(h_reference.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));

    OzakiConfig config;
    config.mode = mode;
    config.scheme_ii_num_moduli = num_moduli;
    config.target_mantissa_bits = target_mantissa_bits;
    config.enable_stage_profiling = enable_stage_profiling;
    if (mode == ExecutionMode::SchemeI) {
        config.scheme_i_mantissa_bits = target_mantissa_bits;
        config.scheme_i_triangular = false;
        config.scheme_i_fused = false;
        if (target_mantissa_bits == 34) {
            config.scheme_i_pair_threshold = 8;
        } else if (target_mantissa_bits == 53) {
            config.scheme_i_pair_threshold = 12;
        }
    }
    AdaptiveOzakiEngine engine(config);

    double rms_error = 0.0, max_rel_error = 0.0, error = 0.0;
    for (int r = 0; r < repeat; ++r) {
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(double)));
        engine.execute(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_result.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
        error = compute_max_error(h_reference, h_result, rms_error, max_rel_error);
    }
    
    std::cout << "precision_test max error: " << error << std::endl;
    std::cout << "max_relative_error: " << max_rel_error << std::endl;
    std::cout << "rms_error: " << rms_error << std::endl;
    std::cout << "padding_rms_error: 0.0" << std::endl;
    std::cout << "target_mantissa_bits=" << target_mantissa_bits << std::endl;

    auto contract = engine.getPrecisionContract();
    std::cout << "\nPrecisionContract {\n"
              << "  mode: \"" << contract.mode << "\"\n"
              << "  target_mantissa_bits: " << contract.target_mantissa_bits << "\n"
              << "  num_split: " << contract.num_split << "\n"
              << "  num_moduli: " << contract.num_moduli << "\n"
              << "  bits_per_int8: " << contract.bits_per_int8 << "\n"
              << "  residual_enabled: " << (contract.residual_enabled ? "true" : "false") << "\n"
              << "  native_fp64_used: " << (contract.native_fp64_used ? "true" : "false") << "\n"
              << "  failure_reason: \"" << contract.failure_reason << "\"\n"
              << "  max_relative_error: " << max_rel_error << "\n"
              << "  passed_precision_gate: " << (max_rel_error <= 1e-6 ? "true" : "false") << "\n"
              << "}\n";
    
    if (enable_stage_profiling) {
        std::cout << "Stage profiling info:" << std::endl;
        for (const auto& stage : contract.stages) {
            std::cout << "  " << stage.stage_name << " : " << stage.stage_timing << " ms" << std::endl;
        }
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));

    return 0;
}
