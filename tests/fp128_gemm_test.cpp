#include "AdaptiveOzaki.h"
#include "Float128.h"
#include "dd_moduli_table.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace ozaki;

static bool same_f128(const Float128& a, const Float128& b) {
    return a.sign == b.sign && a.expo == b.expo && a.mant_lo == b.mant_lo && a.mant_hi == b.mant_hi;
}

static void run_case(int m, int n, int k, Float128 a, Float128 b) {
    std::cout << "Testing " << m << "x" << n << "x" << k << "\n";
    std::vector<Float128> h_A((size_t)m * k, a);
    std::vector<Float128> h_B((size_t)k * n, b);
    std::vector<Float128> h_C((size_t)m * n, Float128::zero());

    Float128 *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, h_A.size() * sizeof(Float128));
    cudaMalloc(&dB, h_B.size() * sizeof(Float128));
    cudaMalloc(&dC, h_C.size() * sizeof(Float128));
    cudaMemcpy(dA, h_A.data(), h_A.size() * sizeof(Float128), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, h_B.data(), h_B.size() * sizeof(Float128), cudaMemcpyHostToDevice);

    OzakiConfig config;
    config.mode = ExecutionMode::SchemeII;
    config.precision_policy = PrecisionPolicy::Auto;
    AdaptiveOzakiEngine engine(config);
    engine.allocateWorkspace(m, n, k);
    engine.execute(dA, dB, dC, m, n, k);

    auto contract = engine.getPrecisionContract();
    std::cout << "contract input_type=" << contract.input_type
              << " target_mantissa_bits=" << contract.target_mantissa_bits
              << " num_moduli=" << contract.num_moduli
              << " crt_bits=" << contract.crt_bits
              << " required_crt_bits=" << contract.required_crt_bits
              << "\n";

    cudaMemcpy(h_C.data(), dC, h_C.size() * sizeof(Float128), cudaMemcpyDeviceToHost);

    Float128 ref = (a * b) * Float128::from_double((double)k);

    std::cout << "got expo=" << h_C[0].expo << " Mhi=" << h_C[0].mant_hi << " Mlo=" << h_C[0].mant_lo
              << " to_double=" << h_C[0].to_double() << "\n";
    std::cout << "ref expo=" << ref.expo << " Mhi=" << ref.mant_hi << " Mlo=" << ref.mant_lo
              << " to_double=" << ref.to_double() << "\n";

    if (contract.input_type != "FP128" || contract.target_mantissa_bits != 113) {
        std::cerr << "FAIL contract fields\n";
        std::exit(1);
    }
    if (!same_f128(h_C[0], ref)) {
        // allow 1-ulp in exponent-normalized M
        UInt384 g, r;
        bool gn, rn;
        h_C[0].to_scaled(&g, &gn, 120);
        ref.to_scaled(&r, &rn, 120);
        UInt384 diff = (g < r) ? (r - g) : (g - r);
        if (gn != rn || diff.bit_length() > 8) {
            std::cerr << "FAIL mismatch vs Float128 oracle\n";
            std::exit(1);
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main() {
    {
        Float128 one = Float128::from_double(1.0);
        Float128 with_lo = Float128::from_double_plus_pow2(1.0, -110);
        if (one.to_double() != with_lo.to_double()) {
            std::cout << "note: to_double distinguishes 1 and 1+2^-110\n";
        } else {
            std::cout << "to_double collapses 1 and 1+2^-110 (expected)\n";
        }
        if (same_f128(one, with_lo)) {
            std::cerr << "FAIL Float128 lost 2^-110\n";
            return 1;
        }
        std::cout << "scalar: 1+2^-110 is distinct from 1 in Float128\n";
    }

    run_case(1, 1, 1, Float128::from_double_plus_pow2(1.0, -110), Float128::from_double(1.0));
    run_case(1, 1, 1, Float128::from_double_plus_pow2(1.0, -80), Float128::from_double(1.0));
    run_case(2, 2, 3, Float128::from_double(1.0), Float128::from_double(1.0));
    run_case(8, 8, 8, Float128::from_double(2.0), Float128::from_double(3.0));
    run_case(32, 32, 32, Float128::from_double_plus_pow2(1.0, -110), Float128::from_double(1.0));
    run_case(64, 64, 64, Float128::from_double_plus_pow2(1.0, -110), Float128::from_double(1.0));
    run_case(128, 128, 128, Float128::from_double_plus_pow2(1.0, -110), Float128::from_double(1.0));

    std::cout << "Testing Failure Mode K=4096 num_moduli=30\n";
    {
        OzakiConfig config;
        config.mode = ExecutionMode::SchemeII;
        config.precision_policy = PrecisionPolicy::Fixed;
        config.scheme_ii_num_moduli = 30;
        config.allow_native_fp64_fallback = false;
        AdaptiveOzakiEngine engine(config);
        engine.allocateWorkspace(1, 1, 4096);
        Float128 *dA, *dB, *dC;
        cudaMalloc(&dA, 4096 * sizeof(Float128));
        cudaMalloc(&dB, 4096 * sizeof(Float128));
        cudaMalloc(&dC, sizeof(Float128));
        try {
            engine.execute(dA, dB, dC, 1, 1, 4096);
            std::cerr << "FAIL: 30 moduli at K=4096 did not throw\n";
            return 1;
        } catch (const std::exception& e) {
            std::cout << "caught: " << e.what() << "\n";
            auto c = engine.getPrecisionContract();
            if (c.failure_reason.find("Insufficient CRT bits") == std::string::npos) {
                std::cerr << "FAIL unexpected reason: " << c.failure_reason << "\n";
                return 1;
            }
        }
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    std::cout << "All FP128 tests completed.\n";
    return 0;
}
