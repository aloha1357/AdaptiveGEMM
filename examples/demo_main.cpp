#include "AdaptiveOzaki.h"
#include "int8_gemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#ifdef _WIN32
#include <stdlib.h>
#endif

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

static double time_ms(int runs, const std::function<void()>& fn) {
    for (int i = 0; i < 2; ++i) fn();
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) fn();
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / runs;
}

int main() {
    const int n = 2048;
    const int element_count = n * n;
    const int runs = 5;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "  SM: " << prop.major << "." << prop.minor
              << ", SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "  clockRate: " << clock_khz / 1000 << " MHz\n";
    std::cout << "  memoryBusWidth: " << prop.memoryBusWidth << " bit\n";
    std::cout << "  sharedMemPerMultiprocessor: "
              << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
    std::cout << "  L2: " << prop.l2CacheSize / (1024 * 1024) << " MB\n";
    std::cout << "  totalGlobalMem: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "\n";

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

    // Do not let an inherited env var silently turn native DGEMM into emulation.
    _putenv("CUBLAS_EMULATE_DOUBLE_PRECISION=");

    cublasHandle_t handle_native;
    cublasHandle_t handle_emu;
    CHECK_CUBLAS(cublasCreate(&handle_native));
    CHECK_CUBLAS(cublasCreate(&handle_emu));

    const size_t ws_bytes = 1024ull * 1024ull * 1024ull;
    void* emu_ws = nullptr;
    CHECK_CUDA(cudaMalloc(&emu_ws, ws_bytes));
    CHECK_CUBLAS(cublasSetWorkspace(handle_emu, emu_ws, ws_bytes));
    CHECK_CUBLAS(cublasSetEmulationStrategy(handle_emu, CUBLAS_EMULATION_STRATEGY_EAGER));
    CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaControl(
        handle_emu, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));

    const double alpha = 1.0;
    const double beta = 0.0;

    auto cublas_rowmajor_ab = [&](cublasHandle_t h, cublasComputeType_t compute) {
        CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, CUDA_R_64F, n,
                                  d_A, CUDA_R_64F, n, &beta, d_C, CUDA_R_64F, n, compute,
                                  CUBLAS_GEMM_DEFAULT));
    };

    // Row-major C=A*B is column-major C=B*A with the same pointers.
    const double native_ms =
        time_ms(runs, [&]() { cublas_rowmajor_ab(handle_native, CUBLAS_COMPUTE_64F); });
    CHECK_CUDA(cudaMemcpy(h_C_reference.data(), d_C, element_count * sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::vector<double> h_C_emu(element_count, 0.0);
    const double emu_ms = time_ms(
        runs, [&]() { cublas_rowmajor_ab(handle_emu, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT); });
    CHECK_CUDA(cudaMemcpy(h_C_emu.data(), d_C, element_count * sizeof(double), cudaMemcpyDeviceToHost));

    OzakiConfig config;
    config.mode = ExecutionMode::Phase22;
    config.enable_fp64 = false;
    config.enable_residual = false;
    AdaptiveOzakiEngine engine(config);

    const double adaptive_ms = time_ms(runs, [&]() { engine.execute(d_A, d_B, d_C, n, n, n); });
    CHECK_CUDA(cudaMemcpy(h_C_result.data(), d_C, element_count * sizeof(double), cudaMemcpyDeviceToHost));

    const double flop = 2.0 * n * n * n;
    auto gflops = [&](double ms) { return flop / (ms * 1e-3) / 1e9; };

    std::cout << "2048^3 GEMM  native vs cuBLAS Ozaki emu vs Phase22\n";
    std::cout << "GPU: " << prop.name << "  SM " << prop.major << "." << prop.minor << "\n";
    std::cout << "CUDA runtime: " << CUDART_VERSION << "\n\n";
    std::cout << "cuBLAS native DGEMM:     " << native_ms << " ms  (" << gflops(native_ms)
              << " GFLOPS)\n";
    std::cout << "cuBLAS FP64 emu (Ozaki): " << emu_ms << " ms  (" << gflops(emu_ms) << " GFLOPS)  "
              << (native_ms / emu_ms) << "x vs native  err " << max_error(h_C_reference, h_C_emu)
              << "\n";
    std::cout << "Adaptive Phase22:        " << adaptive_ms << " ms  (" << gflops(adaptive_ms)
              << " GFLOPS)  " << (native_ms / adaptive_ms) << "x vs native  err "
              << max_error(h_C_reference, h_C_result) << "\n";
    std::cout << "Phase22 vs cuBLAS emu:   " << (emu_ms / adaptive_ms) << "x ( >1 means Phase22 faster )\n";

    std::cout << "\ncuBLAS emu fixed mantissa (Phase 22 is ~34-bit Ozaki):\n";
    CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaControl(
        handle_emu, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
    std::vector<double> h_bits(element_count);
    const int bit_counts[] = {16, 24, 34, 53};
    for (int bits : bit_counts) {
        CHECK_CUBLAS(cublasSetFixedPointEmulationMaxMantissaBitCount(handle_emu, bits));
        const double ms = time_ms(
            runs, [&]() { cublas_rowmajor_ab(handle_emu, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT); });
        CHECK_CUDA(cudaMemcpy(h_bits.data(), d_C, element_count * sizeof(double),
                              cudaMemcpyDeviceToHost));
        const int slices = (bits + 1 + 7) / 8;
        std::cout << "  mantissa " << bits << "  (~" << slices << " slices, " << (slices * slices)
                  << " GEMMs): " << ms << " ms  err " << max_error(h_C_reference, h_bits)
                  << "  vs Phase22 " << (adaptive_ms / ms) << "x ( >1 means emu faster )\n";
    }

    double cpu = 0.0;
    for (int kk = 0; kk < n; ++kk) cpu += h_A[0 * n + kk] * h_B[kk * n + 0];
    std::cout << "C[0,0] adaptive=" << h_C_result[0] << "  cpu=" << cpu
              << "  cublas=" << h_C_reference[0] << "  |err|="
              << std::abs(h_C_result[0] - cpu) << '\n';
    const int mid = n / 2;
    cpu = 0.0;
    for (int kk = 0; kk < n; ++kk) cpu += h_A[mid * n + kk] * h_B[kk * n + mid];
    std::cout << "C[mid,mid] adaptive=" << h_C_result[mid * n + mid] << "  cpu=" << cpu
              << "  cublas=" << h_C_reference[mid * n + mid] << "  |err|="
              << std::abs(h_C_result[mid * n + mid] - cpu) << '\n';

    size_t worst = 0;
    double worst_err = 0.0;
    for (size_t i = 0; i < h_C_reference.size(); ++i) {
        double e = std::abs(h_C_reference[i] - h_C_result[i]);
        if (e > worst_err) {
            worst_err = e;
            worst = i;
        }
    }
    int wr = (int)(worst / n), wc = (int)(worst % n);
    std::cout << "worst err " << worst_err << " at (" << wr << "," << wc
              << ") adaptive=" << h_C_result[worst] << " cublas=" << h_C_reference[worst] << '\n';

    {
        constexpr int n8 = 2048;
        std::cout << "\n[INT8 GEMM] benchmarking " << n8 << "^3 vs cublasGemmEx INT8...\n";
        Int8GemmBench b = bench_int8_gemm_vs_cublas(n8, n8, n8);
        std::cout << "[INT8 GEMM] our kernel:    " << b.our_ms << " ms  " << b.our_tops << " TOPS\n";
        std::cout << "[INT8 GEMM] cuBLAS INT8:   " << b.cublas_ms << " ms  " << b.cublas_tops
                  << " TOPS\n";
        std::cout << "[INT8 GEMM] vs 121 TOPS:   " << b.peak_frac * 100.0 << "%\n";
        std::cout << "[INT8 GEMM] vs cuBLAS INT8:" << b.vs_cublas * 100.0 << "%\n";
    }

    {
        OzakiConfig cfg_scheme_i;
        cfg_scheme_i.mode = ExecutionMode::SchemeI;
        cfg_scheme_i.scheme_i_mantissa_bits = 34;
        cfg_scheme_i.scheme_i_triangular = true;
        cfg_scheme_i.scheme_i_pair_threshold = 4;
        cfg_scheme_i.scheme_i_use_cublas_int8 = true;
        AdaptiveOzakiEngine engine_i(cfg_scheme_i);
        std::vector<double> h_C_scheme_i(element_count, 0.0);
        const double scheme_i_ms = time_ms(runs, [&]() { engine_i.execute(d_A, d_B, d_C, n, n, n); });
        CHECK_CUDA(cudaMemcpy(h_C_scheme_i.data(), d_C, element_count * sizeof(double), cudaMemcpyDeviceToHost));
        std::cout << "\n[SchemeI] bits=34 p=5 pairs=15 backend=cublas  total " << scheme_i_ms << " ms  err " << max_error(h_C_reference, h_C_scheme_i) << "\n";
        std::cout << "          (Note: B v0 uses 1760 MiB. 無 6.0 ms 閘)\n";
    }    {
        OzakiConfig cfg_scheme_i;
        cfg_scheme_i.mode = ExecutionMode::SchemeI;
        cfg_scheme_i.scheme_i_mantissa_bits = 34;
        cfg_scheme_i.scheme_i_triangular = true;
        cfg_scheme_i.scheme_i_pair_threshold = 4;
        cfg_scheme_i.scheme_i_use_cublas_int8 = false;
        cfg_scheme_i.scheme_i_fused = false;
        AdaptiveOzakiEngine engine_i(cfg_scheme_i);
        std::vector<double> h_C_scheme_i(element_count, 0.0);
        const double scheme_i_ms = time_ms(runs, [&]() { engine_i.execute(d_A, d_B, d_C, n, n, n); });
        CHECK_CUDA(cudaMemcpy(h_C_scheme_i.data(), d_C, element_count * sizeof(double), cudaMemcpyDeviceToHost));
        std::cout << "[SchemeI] bits=34 p=5 pairs=15 backend=intree_f64_epilogue  total " << scheme_i_ms << " ms  err " << max_error(h_C_reference, h_C_scheme_i) << "\n";
        std::cout << "          (Note: B.1 估 1280 MiB；6.0 ms 是量測閘不是保證。)\n";
    }

    CHECK_CUBLAS(cublasDestroy(handle_native));
    CHECK_CUBLAS(cublasDestroy(handle_emu));
    CHECK_CUDA(cudaFree(emu_ws));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}