#include "AdaptiveOzaki.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace ozaki;

#define CHECK_CUDA(call)                                                                      \
    do {                                                                                      \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess) {                                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;                                       \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

#define CHECK_CUBLAS(call)                                                                    \
    do {                                                                                      \
        cublasStatus_t err = call;                                                            \
        if (err != CUBLAS_STATUS_SUCCESS) {                                                   \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

static void fill_random(std::vector<double>& matrix, double min_value, double max_value) {
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(min_value, max_value);
    for (double& value : matrix) {
        value = distribution(generator);
    }
}

static double max_error(const std::vector<double>& reference, const std::vector<double>& actual) {
    double result = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        result = std::max(result, std::abs(reference[i] - actual[i]));
    }
    return result;
}

int main() {
    const int n = 2048;
    const int element_count = n * n;

    std::vector<double> h_A(element_count);
    std::vector<double> h_B(element_count);
    std::vector<double> h_C_reference(element_count, 0.0);
    std::vector<double> h_C_result(element_count, 0.0);

    fill_random(h_A, -1.0, 1.0);
    fill_random(h_B, -1.0, 1.0);

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, element_count * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, element_count * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, element_count * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), element_count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), element_count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, element_count * sizeof(double)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const double alpha = 1.0;
    const double beta = 0.0;

    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n));
    CHECK_CUDA(cudaMemcpy(h_C_reference.data(), d_C, element_count * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemset(d_C, 0, element_count * sizeof(double)));

    OzakiConfig config;
    config.mode = ExecutionMode::Phase22;
    AdaptiveOzakiEngine engine(config);

    engine.execute(d_A, d_B, d_C, n, n, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int runs = 3;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        engine.execute(d_A, d_B, d_C, n, n, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / runs;
    const double gflops = (2.0 * n * n * n) / (avg_ms * 1e-3) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_C_result.data(), d_C, element_count * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "AdaptiveOzaki demo\n";
    std::cout << "Matrix size: " << n << " x " << n << "\n";
    std::cout << "Average time: " << avg_ms << " ms\n";
    std::cout << "Throughput: " << gflops << " GFLOPS\n";
    std::cout << "Max error vs cuBLAS FP64: " << max_error(h_C_reference, h_C_result) << '\n';

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}