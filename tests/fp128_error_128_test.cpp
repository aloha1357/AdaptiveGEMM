#include "AdaptiveOzaki.h"
#include "Float128.h"
#include "DoubleDouble.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace ozaki;

static void check_cuda(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after " << what << ": " << cudaGetErrorString(err) << "\n";
        std::exit(2);
    }
}

static Float128 f128_neg(Float128 x) {
    if (!x.is_zero()) x.sign ^= 1;
    return x;
}

static double f128_rel(const Float128& got, const Float128& ref) {
    if (ref.is_zero()) return got.is_zero() ? 0.0 : 1.0;
    Float128 d = got + f128_neg(ref);
    const double denom = std::max(std::fabs(ref.to_double()), 1e-300);
    return std::fabs(d.to_double()) / denom;
}

struct Err {
    double max_rel = 0.0;
    int i = 0, j = 0;
};

static Err max_rel_f128(const std::vector<Float128>& got, const std::vector<Float128>& ref, int m, int n) {
    Err e;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const double r = f128_rel(got[i * n + j], ref[i * n + j]);
            if (r > e.max_rel) {
                e.max_rel = r;
                e.i = i;
                e.j = j;
            }
        }
    }
    return e;
}

static std::vector<Float128> cpu_f128_gemm(const std::vector<Float128>& A,
                                           const std::vector<Float128>& B,
                                           int m, int n, int k) {
    std::vector<Float128> C((size_t)m * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Float128 acc = Float128::zero();
            for (int t = 0; t < k; ++t) acc = acc + (A[i * k + t] * B[t * n + j]);
            C[i * n + j] = acc;
        }
    }
    return C;
}

static std::vector<double> cpu_fp64_gemm(const std::vector<Float128>& A,
                                         const std::vector<Float128>& B,
                                         int m, int n, int k) {
    std::vector<double> C((size_t)m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int t = 0; t < k; ++t) acc += A[i * k + t].to_double() * B[t * n + j].to_double();
            C[i * n + j] = acc;
        }
    }
    return C;
}

static std::vector<DoubleDouble> cpu_dd_gemm(const std::vector<Float128>& A,
                                            const std::vector<Float128>& B,
                                            int m, int n, int k) {
    std::vector<DoubleDouble> C((size_t)m * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            DoubleDouble acc(0.0, 0.0);
            for (int t = 0; t < k; ++t) {
                const double ah = A[i * k + t].to_double();
                const double bh = B[t * n + j].to_double();
                acc = acc + DoubleDouble::from(ah * bh, 0.0);
            }
            C[i * n + j] = acc;
        }
    }
    return C;
}

static Err max_rel_fp64(const std::vector<double>& got, const std::vector<Float128>& ref, int m, int n) {
    Err e;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const Float128 g = Float128::from_double(got[i * n + j]);
            const double r = f128_rel(g, ref[i * n + j]);
            if (r > e.max_rel) {
                e.max_rel = r;
                e.i = i;
                e.j = j;
            }
        }
    }
    return e;
}

static Err max_rel_dd(const std::vector<DoubleDouble>& got, const std::vector<Float128>& ref, int m, int n) {
    Err e;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const Float128 g = Float128::from_double(got[i * n + j].hi) +
                               Float128::from_double(got[i * n + j].lo);
            const double r = f128_rel(g, ref[i * n + j]);
            if (r > e.max_rel) {
                e.max_rel = r;
                e.i = i;
                e.j = j;
            }
        }
    }
    return e;
}

static std::vector<Float128> run_engine(const std::vector<Float128>& A,
                                        const std::vector<Float128>& B,
                                        int m, int n, int k,
                                        PrecisionContract* contract) {
    Float128 *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, A.size() * sizeof(Float128));
    cudaMalloc(&dB, B.size() * sizeof(Float128));
    cudaMalloc(&dC, (size_t)m * n * sizeof(Float128));
    check_cuda("malloc");
    cudaMemcpy(dA, A.data(), A.size() * sizeof(Float128), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * sizeof(Float128), cudaMemcpyHostToDevice);

    OzakiConfig config;
    config.mode = ExecutionMode::SchemeII;
    config.precision_policy = PrecisionPolicy::Auto;
    AdaptiveOzakiEngine engine(config);
    engine.allocateWorkspace(m, n, k);
    engine.execute(dA, dB, dC, m, n, k);
    check_cuda("execute");
    if (contract) *contract = engine.getPrecisionContract();

    std::vector<Float128> C((size_t)m * n);
    cudaMemcpy(C.data(), dC, C.size() * sizeof(Float128), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return C;
}

static bool run_case(const std::string& name,
                     const std::vector<Float128>& A,
                     const std::vector<Float128>& B,
                     int m, int n, int k,
                     double fp128_rel_limit) {
    std::cout << "\n=== " << name << " " << m << "x" << n << "x" << k << " ===\n";
    PrecisionContract contract;
    const std::vector<Float128> got = run_engine(A, B, m, n, k, &contract);
    std::cout << "contract input_type=" << contract.input_type
              << " num_moduli=" << contract.num_moduli
              << " crt_bits=" << contract.crt_bits
              << " required_crt_bits=" << contract.required_crt_bits
              << " native_fp64_used=" << (contract.native_fp64_used ? "true" : "false")
              << "\n";

    std::vector<Float128> ref;
    if (name.find("cancellation") != std::string::npos) {
        ref.assign((size_t)m * n, Float128::from_pow2(-110) * Float128::from_double((double)k));
    } else if (name.find("constant") != std::string::npos) {
        ref.assign((size_t)m * n, (A[0] * B[0]) * Float128::from_double((double)k));
    } else {
        ref = cpu_f128_gemm(A, B, m, n, k);
    }
    const std::vector<double> fp64 = cpu_fp64_gemm(A, B, m, n, k);
    const std::vector<DoubleDouble> dd = cpu_dd_gemm(A, B, m, n, k);

    std::cout << "C[0,0] got=" << got[0].to_double() << " expo=" << got[0].expo
              << " Mlo=" << got[0].mant_lo << "\n";
    std::cout << "C[0,0] ref=" << ref[0].to_double() << " expo=" << ref[0].expo
              << " Mlo=" << ref[0].mant_lo << "\n";
    std::cout << "C[0,0] fp64=" << fp64[0] << " dd=" << (dd[0].hi + dd[0].lo) << "\n";

    const Err e128 = max_rel_f128(got, ref, m, n);
    const Err e64 = max_rel_fp64(fp64, ref, m, n);
    const Err edd = max_rel_dd(dd, ref, m, n);
    std::cout << std::scientific
              << "FP128-vs-oracle max_rel=" << e128.max_rel << " at (" << e128.i << "," << e128.j << ")\n"
              << "FP64-collapsed-vs-oracle max_rel=" << e64.max_rel << "\n"
              << "DD-from-double-vs-oracle max_rel=" << edd.max_rel << "\n";

    bool ok = true;
    if (contract.input_type != "FP128") {
        std::cerr << "FAIL input_type\n";
        ok = false;
    }
    if (e128.max_rel > fp128_rel_limit) {
        std::cerr << "FAIL FP128 max_rel " << e128.max_rel << " > " << fp128_rel_limit << "\n";
        ok = false;
    }
    if (e64.max_rel > 1e-20 && !(e128.max_rel < e64.max_rel)) {
        std::cerr << "FAIL FP128 is not more accurate than FP64 collapse\n";
        ok = false;
    }
    if (e64.max_rel <= 1e-20) {
        const bool extra_bits = got[0].mant_lo != 0 || got[0].mant_hi != Float128::from_double(got[0].to_double()).mant_hi;
        if (!extra_bits && name.find("constant") != std::string::npos) {
            std::cerr << "FAIL constant case lost bits below double\n";
            ok = false;
        }
        if (got[0].mant_lo != ref[0].mant_lo || got[0].mant_hi != ref[0].mant_hi || got[0].expo != ref[0].expo) {
            std::cerr << "FAIL constant/exact M mismatch vs K*(a*b)\n";
            ok = false;
        }
    }
    if (ok) std::cout << "PASS " << name << "\n";
    return ok;
}

int main() {
    const int m = 128, n = 128, k = 128;
    bool all_ok = true;

    {
        std::vector<Float128> A((size_t)m * k);
        std::vector<Float128> B((size_t)k * n, Float128::from_double(1.0));
        for (int i = 0; i < m; ++i) {
            for (int t = 0; t < k; ++t) {
                const double hi = (t % 2 == 0) ? 1.0 : -1.0;
                A[i * k + t] = Float128::from_double_plus_pow2(hi, -110);
            }
        }
        all_ok &= run_case("cancellation_2^-110", A, B, m, n, k, 1e-12);
    }

    {
        std::vector<Float128> A((size_t)m * k, Float128::from_double_plus_pow2(1.0, -110));
        std::vector<Float128> B((size_t)k * n, Float128::from_double(1.0));
        all_ok &= run_case("constant_1+2^-110", A, B, m, n, k, 1e-12);
    }

    {
        std::vector<Float128> A((size_t)m * k);
        std::vector<Float128> B((size_t)k * n);
        for (int i = 0; i < m * k; ++i) {
            A[i] = Float128::from_double(std::sin(0.017 * (i + 1))) + Float128::from_pow2(-110);
        }
        for (int i = 0; i < k * n; ++i) {
            B[i] = Float128::from_double(std::cos(0.013 * (i + 3))) + Float128::from_pow2(-108);
        }
        all_ok &= run_case("varied_hi_plus_2^-110", A, B, m, n, k, 1e-10);
    }

    if (!all_ok) {
        std::cerr << "\n128^3 FP128 error verification FAILED\n";
        return 1;
    }
    std::cout << "\n128^3 FP128 error verification PASSED\n";
    return 0;
}
