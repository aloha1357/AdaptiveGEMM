#include "AdaptiveOzaki.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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
    std::mt19937 generator(123);
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
    const int n = 512;
    const int element_count = n * n;
    const double tolerance = 1e-8;

    std::vector<double> h_A(element_count);
    std::vector<double> h_B(element_count);
    std::vector<double> h_reference(element_count, 0.0);
    std::vector<double> h_result(element_count, 0.0);

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
    CHECK_CUDA(cudaMemcpy(h_reference.data(), d_C, element_count * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemset(d_C, 0, element_count * sizeof(double)));

    OzakiConfig config;
    config.mode = ExecutionMode::Phase22;
    AdaptiveOzakiEngine engine(config);
    engine.execute(d_A, d_B, d_C, n, n, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_C, element_count * sizeof(double), cudaMemcpyDeviceToHost));

    const double error = max_error(h_reference, h_result);
    std::cout << "precision_test max error: " << error << std::endl;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    if (error > tolerance) {
        std::cerr << "precision test failed: max error " << error << " exceeds tolerance " << tolerance << std::endl;
        return 1;
    }

    return 0;
}
