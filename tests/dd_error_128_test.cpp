#include "AdaptiveOzaki.h"
#include "DoubleDouble.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
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

static DoubleDouble dd_mul(const DoubleDouble& a, const DoubleDouble& b) {
    const double p = a.hi * b.hi;
    const double e = std::fma(a.hi, b.hi, -p);
    const double t = std::fma(a.hi, b.lo, std::fma(a.lo, b.hi, a.lo * b.lo));
    return DoubleDouble::from(p, e + t);
}

static double dd_mag(const DoubleDouble& x) {
    return std::abs(x.hi) + std::abs(x.lo);
}

static double dd_abs_diff(const DoubleDouble& a, const DoubleDouble& b) {
    const DoubleDouble d = a + DoubleDouble(-b.hi, -b.lo);
    return std::abs(d.hi) + std::abs(d.lo);
}

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    int worst_i = 0;
    int worst_j = 0;
    DoubleDouble worst_got;
    DoubleDouble worst_ref;
};

static ErrorStats compare_all(const std::vector<DoubleDouble>& got,
                              const std::vector<DoubleDouble>& ref,
                              int m, int n) {
    ErrorStats st;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const int idx = i * n + j;
            const double abs_e = dd_abs_diff(got[idx], ref[idx]);
            const double denom = std::max(dd_mag(ref[idx]), 1e-30);
            const double rel_e = abs_e / denom;
            if (abs_e > st.max_abs) st.max_abs = abs_e;
            if (rel_e > st.max_rel) {
                st.max_rel = rel_e;
                st.worst_i = i;
                st.worst_j = j;
                st.worst_got = got[idx];
                st.worst_ref = ref[idx];
            }
        }
    }
    return st;
}

static ErrorStats compare_fp64(const std::vector<double>& got,
                               const std::vector<DoubleDouble>& ref,
                               int m, int n) {
    ErrorStats st;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const int idx = i * n + j;
            const DoubleDouble g(got[idx], 0.0);
            const double abs_e = dd_abs_diff(g, ref[idx]);
            const double denom = std::max(dd_mag(ref[idx]), 1e-30);
            const double rel_e = abs_e / denom;
            if (abs_e > st.max_abs) st.max_abs = abs_e;
            if (rel_e > st.max_rel) {
                st.max_rel = rel_e;
                st.worst_i = i;
                st.worst_j = j;
                st.worst_got = g;
                st.worst_ref = ref[idx];
            }
        }
    }
    return st;
}

static std::vector<DoubleDouble> cpu_dd_gemm(const std::vector<DoubleDouble>& A,
                                             const std::vector<DoubleDouble>& B,
                                             int m, int n, int k) {
    std::vector<DoubleDouble> C(static_cast<size_t>(m) * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            DoubleDouble acc(0.0, 0.0);
            for (int t = 0; t < k; ++t) {
                acc = acc + dd_mul(A[i * k + t], B[t * n + j]);
            }
            C[i * n + j] = acc;
        }
    }
    return C;
}

static std::vector<double> cpu_fp64_gemm(const std::vector<DoubleDouble>& A,
                                         const std::vector<DoubleDouble>& B,
                                         int m, int n, int k) {
    std::vector<double> C(static_cast<size_t>(m) * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int t = 0; t < k; ++t) {
                acc += (A[i * k + t].hi + A[i * k + t].lo) * (B[t * n + j].hi + B[t * n + j].lo);
            }
            C[i * n + j] = acc;
        }
    }
    return C;
}

static double elapsed_ms(std::chrono::steady_clock::time_point t0,
                         std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct EngineRun {
    std::vector<DoubleDouble> C;
    PrecisionContract contract;
    double execute_ms = 0.0;
};

static EngineRun run_engine(const std::vector<DoubleDouble>& h_A,
                            const std::vector<DoubleDouble>& h_B,
                            int m, int n, int k,
                            int timed_repeats) {
    DoubleDouble *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, static_cast<size_t>(m) * k * sizeof(DoubleDouble));
    cudaMalloc(&d_B, static_cast<size_t>(k) * n * sizeof(DoubleDouble));
    cudaMalloc(&d_C, static_cast<size_t>(m) * n * sizeof(DoubleDouble));
    check_cuda("malloc");

    cudaMemcpy(d_A, h_A.data(), static_cast<size_t>(m) * k * sizeof(DoubleDouble), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), static_cast<size_t>(k) * n * sizeof(DoubleDouble), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, static_cast<size_t>(m) * n * sizeof(DoubleDouble));

    OzakiConfig config;
    config.precision_policy = PrecisionPolicy::Auto;
    config.mode = ExecutionMode::SchemeII;
    AdaptiveOzakiEngine engine(config);
    engine.allocateWorkspace(m, n, k);

    engine.execute(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    check_cuda("warmup execute");

    const int reps = std::max(timed_repeats, 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        engine.execute(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    EngineRun out;
    out.contract = engine.getPrecisionContract();
    out.execute_ms = elapsed_ms(t0, t1) / reps;
    out.C.resize(static_cast<size_t>(m) * n);
    cudaMemcpy(out.C.data(), d_C, static_cast<size_t>(m) * n * sizeof(DoubleDouble), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return out;
}

static void print_stats(const char* name, const ErrorStats& st) {
    std::cout << std::scientific << std::setprecision(6)
              << name
              << " max_abs=" << st.max_abs
              << " max_rel=" << st.max_rel
              << " worst=(" << st.worst_i << "," << st.worst_j << ")"
              << " got(hi,lo)=(" << st.worst_got.hi << "," << st.worst_got.lo << ")"
              << " ref(hi,lo)=(" << st.worst_ref.hi << "," << st.worst_ref.lo << ")"
              << "\n";
}

static int count_collapsed(const std::vector<DoubleDouble>& xs) {
    int n = 0;
    for (const auto& x : xs) {
        if ((x.hi + x.lo) == x.hi) ++n;
    }
    return n;
}

static bool run_case(const std::string& name,
                     const std::vector<DoubleDouble>& A,
                     const std::vector<DoubleDouble>& B,
                     int m, int n, int k,
                     double dd_rel_limit) {
    std::cout << "\n=== " << name << "  " << m << "x" << n << "x" << k << " ===\n";
    std::cout << "A collapsed (hi+lo)==hi: " << count_collapsed(A) << "/" << (m * k) << "\n";
    std::cout << "B collapsed (hi+lo)==hi: " << count_collapsed(B) << "/" << (k * n) << "\n";

    auto t0 = std::chrono::steady_clock::now();
    const std::vector<DoubleDouble> ref = cpu_dd_gemm(A, B, m, n, k);
    auto t1 = std::chrono::steady_clock::now();
    const double cpu_dd_ms = elapsed_ms(t0, t1);

    t0 = std::chrono::steady_clock::now();
    const std::vector<double> fp64 = cpu_fp64_gemm(A, B, m, n, k);
    t1 = std::chrono::steady_clock::now();
    const double cpu_fp64_ms = elapsed_ms(t0, t1);

    const EngineRun engine = run_engine(A, B, m, n, k, 3);
    const std::vector<DoubleDouble>& got = engine.C;
    const PrecisionContract& contract = engine.contract;

    std::cout << "contract input_type=" << contract.input_type
              << " num_moduli=" << contract.num_moduli
              << " crt_bits=" << contract.crt_bits
              << " required_crt_bits=" << contract.required_crt_bits
              << " failure_reason=\"" << contract.failure_reason << "\""
              << " native_fp64_used=" << (contract.native_fp64_used ? "true" : "false")
              << "\n";
    std::cout << std::fixed << std::setprecision(3)
              << "timing_ms cpu_dd_oracle=" << cpu_dd_ms
              << " cpu_fp64=" << cpu_fp64_ms
              << " engine_dd_avg3=" << engine.execute_ms
              << "\n"
              << std::scientific << std::setprecision(6);
    std::cout << "C[0,0] got hi=" << got[0].hi << " lo=" << got[0].lo << "\n";
    std::cout << "C[0,0] ref hi=" << ref[0].hi << " lo=" << ref[0].lo << "\n";
    std::cout << "C[0,0] fp64=" << fp64[0] << "\n";

    const ErrorStats dd_st = compare_all(got, ref, m, n);
    const ErrorStats fp_st = compare_fp64(fp64, ref, m, n);
    print_stats("DoubleDouble-vs-oracle", dd_st);
    print_stats("FP64-collapsed-vs-oracle", fp_st);

    bool ok = true;
    if (dd_st.max_rel > dd_rel_limit) {
        std::cerr << "FAIL: DoubleDouble max_rel " << dd_st.max_rel
                  << " exceeds limit " << dd_rel_limit << "\n";
        ok = false;
    }
    if (!(dd_st.max_abs < fp_st.max_abs) && fp_st.max_abs > 0.0) {
        std::cerr << "FAIL: DoubleDouble is not more accurate than FP64-collapsed "
                  << "(abs " << dd_st.max_abs << " vs " << fp_st.max_abs << ")\n";
        ok = false;
    }
    if (ok) std::cout << "PASS " << name << "\n";
    return ok;
}

int main() {
    const int m = 128, n = 128, k = 128;
    bool all_ok = true;

    {
        std::vector<DoubleDouble> A(static_cast<size_t>(m) * k);
        std::vector<DoubleDouble> B(static_cast<size_t>(k) * n, DoubleDouble(1.0, 0.0));
        const double eps = std::ldexp(1.0, -60);
        for (int i = 0; i < m; ++i) {
            for (int t = 0; t < k; ++t) {
                const double hi = (t % 2 == 0) ? 1.0 : -1.0;
                A[i * k + t] = DoubleDouble(hi, eps);
            }
        }
        all_ok &= run_case("cancellation_lo_only", A, B, m, n, k, 1e-10);
    }

    {
        std::vector<DoubleDouble> A(static_cast<size_t>(m) * k, DoubleDouble(1.0, 1e-20));
        std::vector<DoubleDouble> B(static_cast<size_t>(k) * n, DoubleDouble(1.0, 1e-20));
        all_ok &= run_case("constant_lo_1e-20", A, B, m, n, k, 1e-18);
    }

    {
        std::vector<DoubleDouble> A(static_cast<size_t>(m) * k);
        std::vector<DoubleDouble> B(static_cast<size_t>(k) * n);
        const double lo_scale = std::ldexp(1.0, -80);
        for (int i = 0; i < m * k; ++i) {
            const double hi = std::sin(0.017 * (i + 1));
            const double lo = lo_scale * ((i % 7) - 3);
            A[i] = DoubleDouble::from(hi, lo);
        }
        for (int i = 0; i < k * n; ++i) {
            const double hi = std::cos(0.013 * (i + 3));
            const double lo = lo_scale * ((i % 5) - 2);
            B[i] = DoubleDouble::from(hi, lo);
        }
        all_ok &= run_case("varied_hi_lo", A, B, m, n, k, 1e-14);
    }

    if (!all_ok) {
        std::cerr << "\n128^3 DoubleDouble error verification FAILED\n";
        return 1;
    }
    std::cout << "\n128^3 DoubleDouble error verification PASSED\n";
    return 0;
}
