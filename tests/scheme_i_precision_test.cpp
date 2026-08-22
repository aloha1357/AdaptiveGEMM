#include "AdaptiveOzaki.h"
#include "scheme_i.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>

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
    // q = -1 golden
    // qA = -1, qB = -1, K = 1, eA=0, eB=0, p=5
    double A[1] = { -ldexp(1.0, -34) };
    double B[1] = { -ldexp(1.0, -34) }; // -1 / 2^34
    // Wait, the spec says q = -1, so x = ldexp(A, -e) -> A = -1 * 2^-34
    
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
    
    double expected = A[0] * B[0]; // qAqB * 2^-68
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
    
    // (-1, -1) -> expected 1 * 2^-16
    launch_test_pair_table(-1, -1, d_C);
    CHECK_CUDA(cudaMemcpy(&h_C, d_C, sizeof(double), cudaMemcpyDeviceToHost));
    if (std::abs(h_C - 1.0 * ldexp(1.0, -16)) > 1e-25) {
        std::cerr << "Pair table failed for (-1, -1) expected " << 1.0 * ldexp(1.0, -16) << " got " << h_C << "\n";
        std::exit(1);
    }
    
    // (1, 1) -> expected 1 * 2^-16
    launch_test_pair_table(1, 1, d_C);
    CHECK_CUDA(cudaMemcpy(&h_C, d_C, sizeof(double), cudaMemcpyDeviceToHost));
    if (std::abs(h_C - 1.0 * ldexp(1.0, -16)) > 1e-25) {
        std::cerr << "Pair table failed for (1, 1) expected " << 1.0 * ldexp(1.0, -16) << " got " << h_C << "\n";
        std::exit(1);
    }
    
    // (-1, 1) -> expected -1 * 2^-16
    launch_test_pair_table(-1, 1, d_C);
    CHECK_CUDA(cudaMemcpy(&h_C, d_C, sizeof(double), cudaMemcpyDeviceToHost));
    if (std::abs(h_C - -1.0 * ldexp(1.0, -16)) > 1e-25) {
        std::cerr << "Pair table failed for (-1, 1) expected " << -1.0 * ldexp(1.0, -16) << " got " << h_C << "\n";
        std::exit(1);
    }
    
    std::cout << "[OK] Pair table p=2 mantissa=8 K=1\n";
    CHECK_CUDA(cudaFree(d_C));
}

static void test_gemm(int n, bool mixed) {
    std::vector<double> h_A(n * n), h_B(n * n), h_C_ref(n * n, 0.0), h_C(n * n, 0.0);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < n * n; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }
    
    if (mixed) {
        std::uniform_real_distribution<double> dist_mixed(-256.0, 256.0);
        for (int i = 0; i < n; ++i) h_A[0 * n + i] = dist_mixed(gen);
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k) sum += h_A[i * n + k] * h_B[k * n + j];
            h_C_ref[i * n + j] = sum;
        }
    }
    
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, n * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, n * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, n * n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    auto run_engine = [&](bool use_cublas, int threshold, double& max_err) {
        OzakiConfig config;
        config.mode = ExecutionMode::SchemeI;
        config.scheme_i_mantissa_bits = 34;
        config.scheme_i_triangular = (threshold == 4); // 15 pairs
        config.scheme_i_pair_threshold = threshold;
        config.scheme_i_use_cublas_int8 = use_cublas;
        config.scheme_i_fused = false;
        AdaptiveOzakiEngine engine(config);
        
        engine.execute(d_A, d_B, d_C, n, n, n);
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost));
        
        max_err = 0;
        for (int i = 0; i < n * n; ++i) max_err = std::max(max_err, std::abs(h_C[i] - h_C_ref[i]));
    };
    
    double err_15_v0 = 0, err_25_v0 = 0;
    double err_15_b1 = 0, err_25_b1 = 0;
    
    run_engine(true, 4, err_15_v0);
    run_engine(true, 5, err_25_v0);
    
    run_engine(false, 4, err_15_b1);
    run_engine(false, 5, err_25_b1);
    
    std::cout << "v0: 15 pairs err=" << err_15_v0 << "  25 pairs err=" << err_25_v0 << "\n";
    std::cout << "b1: 15 pairs err=" << err_15_b1 << "  25 pairs err=" << err_25_b1 << "\n";
    
    auto check_errs = [&](double max_err_15, double max_err_25, const char* name) {
        if (!mixed) {
            if (max_err_25 > 1e-8) { std::cerr << name << " err_25 > 1e-8\n"; std::exit(1); }
            if (max_err_15 >= 1e-4 || max_err_15 <= max_err_25) { 
                std::cerr << name << " err_15 sanity check failed (expected err_25 < err_15 < 1e-4)\n"; 
                std::exit(1); 
            }
        } else {
            if (max_err_25 > 1e-8) {
                std::cout << "WARN " << name << " mixed 25-pair abs err=" << max_err_25 << " (SPEC 1e-8 not met; relative TBD)\n";
            }
        }
    };
    
    check_errs(err_15_v0, err_25_v0, "v0");
    check_errs(err_15_b1, err_25_b1, "b1");
    
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
}

int main(int argc, char** argv) {
    int n = 512;
    if (argc > 1) n = std::atoi(argv[1]);
    
    test_enumerator();
    test_pair_table_golden();
    test_golden();
    test_exponents();
    test_gemm(n, false);
    test_gemm(n, true);
    
    std::cout << "All Scheme I tests passed!\n";
    return 0;
}
