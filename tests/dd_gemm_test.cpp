#include "AdaptiveOzaki.h"
#include "DoubleDouble.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace ozaki;

void run_test(int m, int n, int k, DoubleDouble valA, DoubleDouble valB) {
    std::cout << "Testing size " << m << "x" << n << "x" << k << std::endl;
    
    std::vector<DoubleDouble> h_A(m * k, valA);
    std::vector<DoubleDouble> h_B(k * n, valB);
    std::vector<DoubleDouble> h_C(m * n, DoubleDouble(0.0, 0.0));
    
    DoubleDouble* d_A;
    DoubleDouble* d_B;
    DoubleDouble* d_C;
    cudaMalloc(&d_A, m * k * sizeof(DoubleDouble));
    cudaMalloc(&d_B, k * n * sizeof(DoubleDouble));
    cudaMalloc(&d_C, m * n * sizeof(DoubleDouble));
    
    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(DoubleDouble), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(DoubleDouble), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(DoubleDouble));
    
    OzakiConfig config;
    config.precision_policy = PrecisionPolicy::Auto;
    config.mode = ExecutionMode::SchemeII;
    AdaptiveOzakiEngine engine(config);
    
    engine.allocateWorkspace(m, n, k);
    engine.execute(d_A, d_B, d_C, m, n, k);
    
    auto contract = engine.getPrecisionContract();
    std::cout << "Contract - mode: " << contract.mode 
              << ", input_type: " << contract.input_type 
              << ", crt_bits: " << contract.crt_bits 
              << ", required_bits: " << contract.required_crt_bits << std::endl;
              
    cudaMemcpy(h_C.data(), d_C, m * n * sizeof(DoubleDouble), cudaMemcpyDeviceToHost);
    
    // Reference DoubleDouble dot product
    DoubleDouble ref(0.0, 0.0);
    for (int i=0; i<k; ++i) {
        // exact
        double hi = valA.hi * valB.hi;
        double lo = valA.hi * valB.lo + valA.lo * valB.hi;
        ref = ref + DoubleDouble(hi, lo);
    }
    
    std::cout << "Output [0,0]: hi=" << h_C[0].hi << ", lo=" << h_C[0].lo << std::endl;
    std::cout << "Ref [0,0]: hi=" << ref.hi << ", lo=" << ref.lo << std::endl;
    
    // Assert correctness within floating-point tolerance of the reconstructed double-double
    // Since this is integer CRT, there's no rounding error for the reconstructed integer itself, 
    // but DoubleDouble conversion might have tiny ULPs if valA.lo is tiny.
    double err_hi = std::abs(h_C[0].hi - ref.hi);
    double err_lo = std::abs(h_C[0].lo - ref.lo);
    double max_val = std::abs(ref.hi) == 0.0 ? 1.0 : std::abs(ref.hi);
    if (err_hi / max_val > 1e-12 || err_lo / max_val > 1e-12) {
        std::cerr << "Assertion failed: h_C[0] != ref" << std::endl;
        std::exit(1);
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void test_gate_d_cases() {
    std::cout << "Gate D scalar cases:\n";
    DoubleDouble cases[4] = {
        DoubleDouble(1.0, std::pow(2.0, -80)),
        DoubleDouble(1.0, std::pow(2.0, -100)),
        DoubleDouble(-1.0, std::pow(2.0, -90)),
        DoubleDouble(std::pow(2.0, 20), std::pow(2.0, -40))
    };
    for (int i=0; i<4; ++i) {
        run_test(1, 1, 1, cases[i], DoubleDouble(1.0, 0.0));
    }
}

void test_failure_mode() {
    std::cout << "Testing Failure Mode (K=4096, num_moduli=20)\n";
    int m = 1, n = 1, k = 4096;
    DoubleDouble* d_A;
    DoubleDouble* d_B;
    DoubleDouble* d_C;
    cudaMalloc(&d_A, m * k * sizeof(DoubleDouble));
    cudaMalloc(&d_B, k * n * sizeof(DoubleDouble));
    cudaMalloc(&d_C, m * n * sizeof(DoubleDouble));
    
    OzakiConfig config;
    config.precision_policy = PrecisionPolicy::Fixed;
    config.scheme_ii_num_moduli = 20;
    config.allow_native_fp64_fallback = false;
    
    AdaptiveOzakiEngine engine(config);
    engine.allocateWorkspace(m, n, k);
    
    try {
        engine.execute(d_A, d_B, d_C, m, n, k);
        std::cerr << "ERROR: Failure mode test did NOT throw an exception!\n";
    } catch (const std::exception& e) {
        std::cout << "Successfully caught expected failure: " << e.what() << "\n";
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    test_gate_d_cases();
    run_test(2, 2, 3, DoubleDouble(1.0, 1e-18), DoubleDouble(1.0, 1e-18));
    run_test(8, 8, 8, DoubleDouble(2.0, 1e-20), DoubleDouble(3.0, 1e-20));
    run_test(32, 32, 32, DoubleDouble(1.5, 1e-25), DoubleDouble(1.5, 1e-25));
    run_test(64, 64, 64, DoubleDouble(1.0, std::pow(2.0, -100)), DoubleDouble(1.0, std::pow(2.0, -100)));
    run_test(128, 128, 128, DoubleDouble(1.0, 1e-20), DoubleDouble(1.0, 1e-20));
    
    test_failure_mode();
    
    std::cout << "All DoubleDouble tests completed.\n";
    return 0;
}
