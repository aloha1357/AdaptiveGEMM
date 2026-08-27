#include "AdaptiveOzaki.h"
#include "scheme_i.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>
#include <cublas_v2.h>

using namespace ozaki;

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(1); } } while (0)
#define CHECK_CUBLAS(call) do { cublasStatus_t err = call; if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(1); } } while (0)

static void test_enumerator() {
    int count15 = scheme_i_for_each_pair(5, true, 4, [](int, int){});
    if (count15 != 15) { std::cerr << "enumerator failed for 15\n"; std::exit(1); }
    int count19 = scheme_i_for_each_pair(5, true, 5, [](int, int){});
    if (count19 != 19) { std::cerr << "enumerator failed for 19\n"; std::exit(1); }
    int count25 = scheme_i_for_each_pair(5, false, 4, [](int, int){});
    if (count25 != 25) { std::cerr << "enumerator failed for 25\n"; std::exit(1); }
    std::cout << "[OK] enumerator\n";
}

static void test_golden() {
    double A[1] = { -ldexp(1.0, -34) };
    double B[1] = { -ldexp(1.0, -34) }; 
    double* d_A; double* d_B; double* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(double), cudaMemcpyHostToDevice));
    
    OzakiConfig config;
    config.mode = ExecutionMode::SchemeI;
    config.scheme_i_mantissa_bits = 34;
    config.scheme_i_triangular = false;
    config.scheme_i_pair_threshold = 4;
    config.scheme_i_use_cublas_int8 = true; 
    AdaptiveOzakiEngine engine(config);
    
    engine.execute(d_A, d_B, d_C, 1, 1, 1);
    
    double C[1];
    CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(double), cudaMemcpyDeviceToHost));
    
    double expected = A[0] * B[0]; 
    if (std::abs(C[0] - expected) > 1e-25) {
        std::cerr << "Golden failed! C=" << C[0] << " expected=" << expected << "\n";
        std::exit(1);
    }
    std::cout << "[OK] Golden q=-1 p=5 K=1\n";
    
    double A2[1] = { ldexp(1.0, -34) };
    double B2[1] = { -ldexp(1.0, -34) };
    CHECK_CUDA(cudaMemcpy(d_A, A2, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B2, sizeof(double), cudaMemcpyHostToDevice));
    engine.execute(d_A, d_B, d_C, 1, 1, 1);
    CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(double), cudaMemcpyDeviceToHost));
    expected = A2[0] * B2[0];
    if (std::abs(C[0] - expected) > 1e-25) {
        std::cerr << "Golden failed! q=1,-1 C=" << C[0] << " expected=" << expected << "\n";
        std::exit(1);
    }
    std::cout << "[OK] Golden q=1,-1 p=5 K=1\n";
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
}

static void test_exponents() {
    double h_B[4 * 1] = { 0.4, 0.0, 0.5, 1.0 };
    double* d_B; double* d_maxB; int32_t* d_e;
    CHECK_CUDA(cudaMalloc(&d_B, 4 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_maxB, 4 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_e, 4 * sizeof(int32_t)));
    
    CHECK_CUDA(cudaMemcpy(d_B, h_B, 4 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_maxB, 0, 4 * sizeof(double)));
    
    launch_scheme_i_reduce_maxB(d_B, d_maxB, 1, 4);
    launch_scheme_i_exponents_from_max(d_maxB, d_e, 4);
    
    int32_t h_e[4];
    CHECK_CUDA(cudaMemcpy(h_e, d_e, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    if (h_e[0] != -1) { std::cerr << "max=0.4 eB expected -1 got " << h_e[0] << "\n"; std::exit(1); }
    if (h_e[1] != 0) { std::cerr << "max=0 eB expected 0 got " << h_e[1] << "\n"; std::exit(1); }
    if (h_e[2] != -1) { std::cerr << "max=0.5 eB expected -1 got " << h_e[2] << "\n"; std::exit(1); }
    if (h_e[3] != 0) { std::cerr << "max=1.0 eB expected 0 got " << h_e[3] << "\n"; std::exit(1); }
    
    std::cout << "[OK] exponents\n";
    CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_maxB)); CHECK_CUDA(cudaFree(d_e));
}

static void test_pair_table_golden() {
    double* d_C;
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(double)));
    double h_C;
    
    launch_test_pair_table(-1, -1, d_C);
    CHECK_CUDA(cudaMemcpy(&h_C, d_C, sizeof(double), cudaMemcpyDeviceToHost));
    if (std::abs(h_C - 1.0 * ldexp(1.0, -16)) > 1e-25) {
        std::cerr << "Pair table failed for (-1, -1) expected " << 1.0 * ldexp(1.0, -16) << " got " << h_C << "\n";
        std::exit(1);
    }
    
    launch_test_pair_table(1, 1, d_C);
    CHECK_CUDA(cudaMemcpy(&h_C, d_C, sizeof(double), cudaMemcpyDeviceToHost));
    if (std::abs(h_C - 1.0 * ldexp(1.0, -16)) > 1e-25) {
        std::cerr << "Pair table failed for (1, 1) expected " << 1.0 * ldexp(1.0, -16) << " got " << h_C << "\n";
        std::exit(1);
    }
    
    launch_test_pair_table(-1, 1, d_C);
    CHECK_CUDA(cudaMemcpy(&h_C, d_C, sizeof(double), cudaMemcpyDeviceToHost));
    if (std::abs(h_C - -1.0 * ldexp(1.0, -16)) > 1e-25) {
        std::cerr << "Pair table failed for (-1, 1) expected " << -1.0 * ldexp(1.0, -16) << " got " << h_C << "\n";
        std::exit(1);
    }
    
    std::cout << "[OK] Pair table p=2 mantissa=8 K=1\n";
    CHECK_CUDA(cudaFree(d_C));
}

static void test_gemm_custom(int M, int N, int K, int repeat, int mantissa_bits = 34, int threshold = 8) {
    std::cout << "\nTesting M=" << M << ", N=" << N << ", K=" << K << ", repeat=" << repeat << ", bits=" << mantissa_bits << ", threshold=" << threshold << "\n";
    std::vector<double> h_A(M * K), h_B(K * N), h_C_ref(M * N, 0.0), h_C(M * N, 0.0);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < M * K; ++i) h_A[i] = dist(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dist(gen);

    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUBLAS(cublasDestroy(handle));

    OzakiConfig config;
    config.mode = ExecutionMode::SchemeI;
    config.scheme_i_mantissa_bits = mantissa_bits;
    config.scheme_i_triangular = false;
    config.scheme_i_pair_threshold = threshold;
    config.scheme_i_fused = false;
    AdaptiveOzakiEngine engine(config);

    double max_err = 0;
    double max_rel_error = 0;
    for (int r = 0; r < repeat; ++r) {
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(double)));
        engine.execute(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < M * N; ++i) {
            double err = std::abs(h_C[i] - h_C_ref[i]);
            max_err = std::max(max_err, err);
            if (std::abs(h_C_ref[i]) > 1e-12) {
                max_rel_error = std::max(max_rel_error, err / std::abs(h_C_ref[i]));
            }
        }
    }
    
    std::cout << "max error: " << max_err << "\n";
    std::cout << "max_relative_error: " << max_rel_error << "\n";
    std::cout << "target_mantissa_bits=" << mantissa_bits << "\n"; 
    std::cout << "target=" << mantissa_bits << "\n";

    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
}


int main(int argc, char** argv) {
    int n = 512;
    if (argc > 1) n = std::atoi(argv[1]);
    
    test_enumerator();
    test_pair_table_golden();
    test_golden();
    test_exponents();
    
    std::cout << "--- Standard tests (34-bit) ---\n";
    test_gemm_custom(n, n, n, 1, 34, 8);
    
    std::cout << "\n--- Rectangular input (M=4096, N=512, K=8192, 34-bit) ---\n";
    test_gemm_custom(4096, 512, 8192, 1, 34, 8);
    
    std::cout << "\n--- Repeated 10 times without degradation (34-bit) ---\n";
    test_gemm_custom(512, 512, 512, 10, 34, 8);
    
    std::cout << "\n--- Standard tests (53-bit) ---\n";
    test_gemm_custom(n, n, n, 1, 53, 12);
    
    std::cout << "\n--- Rectangular input (M=4096, N=512, K=8192, 53-bit) ---\n";
    test_gemm_custom(4096, 512, 8192, 1, 53, 12);
    
    std::cout << "\n--- Repeated 10 times without degradation (53-bit) ---\n";
    test_gemm_custom(512, 512, 512, 10, 53, 12);

    std::cout << "\nAll Scheme I tests passed!\n";
    return 0;
}
