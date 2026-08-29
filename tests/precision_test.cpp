#include "AdaptiveOzaki.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

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

static float time_engine(AdaptiveOzakiEngine& engine, double* d_A, double* d_B, double* d_C,
                         int M, int N, int K, int repeat, cudaEvent_t start, cudaEvent_t stop) {
    engine.execute(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(start);
    for (int r = 0; r < repeat; ++r) {
        engine.execute(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / (float)std::max(repeat, 1);
}

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    ExecutionMode mode = ExecutionMode::Phase22;
    int num_moduli = 15;
    int num_split = 0;
    int bits_per_int8 = 0;
    bool allow_fallback = false;
    int target_mantissa_bits = 53;
    int repeat = 1;
    bool enable_stage_profiling = false;
    bool enable_benchmark = false;
    bool skip_scalA = false;
    bool use_cuda_graph = false;
    bool use_true_fused_kernel = false;
    std::string policy_str = "Fixed";
    std::string scheme_ii_backend_str = "cublas";
    int scheme_ii_group_size = 4;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            std::string mode_str = argv[++i];
            if (mode_str == "Phase22") mode = ExecutionMode::Phase22;
            else if (mode_str == "SchemeI") mode = ExecutionMode::SchemeI;
            else if (mode_str == "SchemeII") mode = ExecutionMode::SchemeII;
            else if (mode_str == "SchemeII_Fused") mode = ExecutionMode::SchemeII_Fused;
            else if (mode_str == "Phase22_Fused") mode = ExecutionMode::Phase22_Fused;
        } else if (arg == "--num_moduli" && i + 1 < argc) {
            num_moduli = std::atoi(argv[++i]);
        } else if (arg == "--num_split" && i + 1 < argc) {
            num_split = std::atoi(argv[++i]);
        } else if (arg == "--bits_per_int8" && i + 1 < argc) {
            bits_per_int8 = std::atoi(argv[++i]);
        } else if (arg == "--allow_fallback") {
            allow_fallback = true;
        } else if (arg == "--precision_policy" && i + 1 < argc) {
            policy_str = argv[++i];
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
        } else if (arg == "--benchmark") {
            enable_benchmark = true;
        } else if (arg == "--stage") {
            enable_stage_profiling = true;
        } else if (arg == "--skip_scalA") {
            skip_scalA = true;
        } else if (arg == "--use_cuda_graph") {
            use_cuda_graph = true;
        } else if (arg == "--use_true_fused_kernel") {
            use_true_fused_kernel = true;
        } else if (arg == "--scheme_ii_backend" && i + 1 < argc) {
            scheme_ii_backend_str = argv[++i];
        } else if (arg == "--scheme_ii_group_size" && i + 1 < argc) {
            scheme_ii_group_size = std::atoi(argv[++i]);
        }
    }

    std::vector<double> h_A(M * K), h_B(K * N), h_reference(M * N, 0.0), h_result(M * N, 0.0);
    fill_random(h_A, -1.0, 1.0);
    fill_random(h_B, -1.0, 1.0);

    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), (size_t)M * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), (size_t)K * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(double)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    const double alpha = 1.0, beta = 0.0;

    cudaEvent_t start_fp64, stop_fp64;
    cudaEventCreate(&start_fp64);
    cudaEventCreate(&stop_fp64);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    cudaEventRecord(start_fp64);
    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    cudaEventRecord(stop_fp64);
    cudaEventSynchronize(stop_fp64);

    float fp64_ms = 0;
    cudaEventElapsedTime(&fp64_ms, start_fp64, stop_fp64);
    std::cout << "Native cuBLAS FP64 Time: " << fp64_ms << " ms\n";

    CHECK_CUDA(cudaMemcpy(h_reference.data(), d_C, (size_t)M * N * sizeof(double), cudaMemcpyDeviceToHost));

    OzakiConfig config;
    config.mode = mode;
    config.scheme_ii_num_moduli = num_moduli;
    config.scheme_i_num_split = num_split;
    config.scheme_i_bits_per_int8 = bits_per_int8;
    config.target_mantissa_bits = target_mantissa_bits;
    config.enable_stage_profiling = enable_stage_profiling;
    config.enable_skip_scalA = skip_scalA;
    config.allow_native_fp64_fallback = allow_fallback;
    config.scheme_ii_use_cuda_graph = use_cuda_graph;
    config.scheme_ii_use_true_fused_kernel = use_true_fused_kernel;
    config.scheme_ii_gemm_backend = (scheme_ii_backend_str == "custom_mma")
        ? SchemeIIGemmBackend::CustomMMA
        : SchemeIIGemmBackend::CuBLAS;
    config.scheme_ii_moduli_group_size = scheme_ii_group_size;

    if (policy_str == "Auto" ||
        (mode == ExecutionMode::SchemeI && num_split == 0) ||
        (mode == ExecutionMode::SchemeII && num_moduli == 0) ||
        (mode == ExecutionMode::SchemeII_Fused && num_moduli == 0)) {
        config.precision_policy = PrecisionPolicy::Auto;
    } else {
        config.precision_policy = PrecisionPolicy::Fixed;
    }

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

    float dyn_emu_ms = -1.0f;
    float fix_emu_ms = -1.0f;
    float schemeII_ms = -1.0f;
    float schemeII_fused_ms = -1.0f;
    float schemeII_fused_cublas_ms = -1.0f;
    float schemeII_fused_custom_mma_ms = -1.0f;

    if (enable_benchmark) {
        cublasHandle_t handle_emu;
        CHECK_CUBLAS(cublasCreate(&handle_emu));

        cudaEventRecord(start_fp64);
        for (int r = 0; r < repeat; r++) {
            cublasGemmEx(handle_emu, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         d_B, CUDA_R_64F, N, d_A, CUDA_R_64F, K, &beta, d_C, CUDA_R_64F, N,
                         CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(stop_fp64);
        cudaEventSynchronize(stop_fp64);
        cudaEventElapsedTime(&dyn_emu_ms, start_fp64, stop_fp64);
        dyn_emu_ms /= (float)std::max(repeat, 1);

        cublasSetFixedPointEmulationMantissaControl(handle_emu, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
        cublasSetFixedPointEmulationMaxMantissaBitCount(handle_emu, 53);

        cudaEventRecord(start_fp64);
        for (int r = 0; r < repeat; r++) {
            cublasGemmEx(handle_emu, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         d_B, CUDA_R_64F, N, d_A, CUDA_R_64F, K, &beta, d_C, CUDA_R_64F, N,
                         CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(stop_fp64);
        cudaEventSynchronize(stop_fp64);
        cudaEventElapsedTime(&fix_emu_ms, start_fp64, stop_fp64);
        fix_emu_ms /= (float)std::max(repeat, 1);
        cublasDestroy(handle_emu);

        OzakiConfig cfg_ii = config;
        cfg_ii.mode = ExecutionMode::SchemeII;
        cfg_ii.scheme_ii_use_cuda_graph = false;
        cfg_ii.enable_stage_profiling = false;
        AdaptiveOzakiEngine engine_ii(cfg_ii);
        schemeII_ms = time_engine(engine_ii, d_A, d_B, d_C, M, N, K, repeat, start_fp64, stop_fp64);
        engine_ii.freeWorkspace();

        const bool want_custom_mma = (config.scheme_ii_gemm_backend == SchemeIIGemmBackend::CustomMMA);

        OzakiConfig cfg_fused_cublas = config;
        cfg_fused_cublas.mode = ExecutionMode::SchemeII_Fused;
        cfg_fused_cublas.scheme_ii_gemm_backend = SchemeIIGemmBackend::CuBLAS;
        AdaptiveOzakiEngine engine_fused_cublas(cfg_fused_cublas);
        schemeII_fused_cublas_ms = time_engine(engine_fused_cublas, d_A, d_B, d_C, M, N, K, repeat, start_fp64, stop_fp64);
        PrecisionContract contract_cublas = engine_fused_cublas.getPrecisionContract();
        if (!want_custom_mma) {
            CHECK_CUDA(cudaMemcpy(h_result.data(), d_C, (size_t)M * N * sizeof(double), cudaMemcpyDeviceToHost));
        }
        engine_fused_cublas.freeWorkspace();

        OzakiConfig cfg_fused_mma = config;
        cfg_fused_mma.mode = ExecutionMode::SchemeII_Fused;
        cfg_fused_mma.scheme_ii_gemm_backend = SchemeIIGemmBackend::CustomMMA;
        AdaptiveOzakiEngine engine_fused_mma(cfg_fused_mma);
        schemeII_fused_custom_mma_ms = time_engine(engine_fused_mma, d_A, d_B, d_C, M, N, K, repeat, start_fp64, stop_fp64);
        PrecisionContract contract_mma = engine_fused_mma.getPrecisionContract();
        if (want_custom_mma) {
            CHECK_CUDA(cudaMemcpy(h_result.data(), d_C, (size_t)M * N * sizeof(double), cudaMemcpyDeviceToHost));
        }

        schemeII_fused_ms = want_custom_mma ? schemeII_fused_custom_mma_ms : schemeII_fused_cublas_ms;
        PrecisionContract contract = want_custom_mma ? contract_mma : contract_cublas;

        double rms_error = 0.0, max_rel_error = 0.0;
        double max_abs_error = compute_max_error(h_reference, h_result, rms_error, max_rel_error);
        bool passed = (max_rel_error <= 1e-6) && !contract.native_fp64_used && !contract.residual_enabled;
        if (want_custom_mma) {
            passed = passed && contract.used_custom_mma && contract.scheme_ii_backend == "custom_mma";
        }

        std::cout << "cuBLAS 13 dynamic emu Time: " << dyn_emu_ms << " ms\n";
        std::cout << "cuBLAS 13 fixed53 emu Time: " << fix_emu_ms << " ms\n";
        std::cout << "SchemeII Time: " << schemeII_ms << " ms\n";
        std::cout << "SchemeII_Fused Time: " << schemeII_fused_ms << " ms\n";
        std::cout << "SchemeII_Fused cuBLAS backend Time: " << schemeII_fused_cublas_ms << " ms\n";
        std::cout << "SchemeII_Fused custom MMA Time: " << schemeII_fused_custom_mma_ms << " ms\n";
        std::cout << "used_cuda_graph=" << (contract.used_cuda_graph ? "true" : "false") << "\n";
        std::cout << "used_true_fused_kernel=" << (contract.used_true_fused_kernel ? "true" : "false") << "\n";
        std::cout << "used_scheme_ii_fused=" << (contract.used_scheme_ii_fused ? "true" : "false") << "\n";
        std::cout << "used_custom_mma=" << (contract.used_custom_mma ? "true" : "false") << "\n";
        std::cout << "scheme_ii_backend=" << (contract.scheme_ii_backend.empty() ? "cublas" : contract.scheme_ii_backend) << "\n";
        std::cout << "scheme_ii_group_size=" << contract.scheme_ii_group_size << "\n";
        std::cout << "scheme_ii_gemm_launches=" << contract.scheme_ii_gemm_launches << "\n";
        std::cout << "used_partial_crt_epilogue=" << (contract.used_partial_crt_epilogue ? "true" : "false") << "\n";
        std::cout << "fallback_mode=" << (contract.fallback_mode.empty() ? "None" : contract.fallback_mode) << "\n";
        std::cout << "num_moduli=" << contract.num_moduli << "\n";
        std::cout << "max_abs_error=" << max_abs_error << "\n";
        std::cout << "max_relative_error=" << max_rel_error << "\n";
        std::cout << "passed_precision_gate=" << (passed ? 1 : 0) << "\n";

        std::cout << "BENCHMARK_RESULT: M=" << M << " N=" << N << " K=" << K
                  << " cublas_fp64_ms=" << fp64_ms
                  << " cublas13_dynamic_emu_ms=" << dyn_emu_ms
                  << " cublas13_fixed53_emu_ms=" << fix_emu_ms
                  << " schemeII_ms=" << schemeII_ms
                  << " schemeII_fused_ms=" << schemeII_fused_ms
                  << " schemeII_fused_cublas_backend_ms=" << schemeII_fused_cublas_ms
                  << " schemeII_fused_custom_mma_ms=" << schemeII_fused_custom_mma_ms
                  << " schemeII_fused_custom_mma_fallback_mode=" << (contract_mma.fallback_mode.empty() ? "None" : contract_mma.fallback_mode)
                  << " schemeII_fused_fallback_mode=" << (contract.fallback_mode.empty() ? "None" : contract.fallback_mode)
                  << " schemeII_fused_num_moduli=" << contract.num_moduli
                  << " fallback_mode=" << (contract.fallback_mode.empty() ? "None" : contract.fallback_mode)
                  << " num_moduli=" << contract.num_moduli
                  << " max_abs_error=" << max_abs_error
                  << " max_relative_error=" << max_rel_error
                  << " passed_precision_gate=" << (passed ? 1 : 0)
                  << " used_cuda_graph=" << (contract.used_cuda_graph ? "true" : "false")
                  << " used_true_fused_kernel=" << (contract.used_true_fused_kernel ? "true" : "false")
                  << " used_custom_mma=" << (contract.used_custom_mma ? "true" : "false")
                  << " scheme_ii_backend=" << (contract.scheme_ii_backend.empty() ? "cublas" : contract.scheme_ii_backend)
                  << " scheme_ii_group_size=" << contract.scheme_ii_group_size
                  << " scheme_ii_gemm_launches=" << contract.scheme_ii_gemm_launches
                  << " used_partial_crt_epilogue=" << (contract.used_partial_crt_epilogue ? "true" : "false")
                  << std::endl;

        engine_fused_mma.freeWorkspace();
        cudaEventDestroy(start_fp64);
        cudaEventDestroy(stop_fp64);
        CHECK_CUBLAS(cublasDestroy(handle));
        CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
        return passed ? 0 : 1;
    }

    double rms_error = 0.0, max_rel_error = 0.0, error = 0.0;
    for (int r = 0; r < repeat; ++r) {
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(double)));
        engine.execute(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_result.data(), d_C, (size_t)M * N * sizeof(double), cudaMemcpyDeviceToHost));
        error = compute_max_error(h_reference, h_result, rms_error, max_rel_error);
    }

    std::cout << "precision_test max error: " << error << std::endl;
    std::cout << "max_abs_error: " << error << std::endl;
    std::cout << "max_relative_error: " << max_rel_error << std::endl;
    std::cout << "rms_error: " << rms_error << std::endl;
    std::cout << "padding_rms_error: 0.0" << std::endl;
    std::cout << "target_mantissa_bits=" << target_mantissa_bits << std::endl;

    auto plan = engine.getSchemePlan();
    std::cout << "\nSchemePlan {\n"
              << "  num_split: " << plan.num_split << "\n"
              << "  bits_per_int8: " << plan.bits_per_int8 << "\n"
              << "  num_moduli: " << plan.num_moduli << "\n"
              << "}\n";

    auto contract = engine.getPrecisionContract();
    bool passed_gate = (max_rel_error <= 1e-6);
    std::cout << "\nPrecisionContract {\n"
              << "  mode: \"" << contract.mode << "\"\n"
              << "  target_mantissa_bits: " << contract.target_mantissa_bits << "\n"
              << "  num_split: " << contract.num_split << "\n"
              << "  num_moduli: " << contract.num_moduli << "\n"
              << "  bits_per_int8: " << contract.bits_per_int8 << "\n"
              << "  residual_enabled: " << (contract.residual_enabled ? "true" : "false") << "\n"
              << "  residual_used: " << (contract.residual_used ? "true" : "false") << "\n"
              << "  native_fp64_used: " << (contract.native_fp64_used ? "true" : "false") << "\n"
              << "  fallback_mode: \"" << (contract.fallback_mode.empty() ? "None" : contract.fallback_mode) << "\"\n"
              << "  failure_reason: \"" << contract.failure_reason << "\"\n"
              << "  max_relative_error: " << max_rel_error << "\n"
              << "  passed_precision_gate: " << (passed_gate ? "true" : "false") << "\n"
              << "  used_cuda_graph: " << (contract.used_cuda_graph ? "true" : "false") << "\n"
              << "  used_true_fused_kernel: " << (contract.used_true_fused_kernel ? "true" : "false") << "\n"
              << "  used_scheme_ii_fused: " << (contract.used_scheme_ii_fused ? "true" : "false") << "\n"
              << "  used_custom_mma: " << (contract.used_custom_mma ? "true" : "false") << "\n"
              << "  scheme_ii_backend: \"" << (contract.scheme_ii_backend.empty() ? "cublas" : contract.scheme_ii_backend) << "\"\n"
              << "  scheme_ii_group_size: " << contract.scheme_ii_group_size << "\n"
              << "  scheme_ii_gemm_launches: " << contract.scheme_ii_gemm_launches << "\n"
              << "  used_partial_crt_epilogue: " << (contract.used_partial_crt_epilogue ? "true" : "false") << "\n"
              << "}\n";

    if (enable_stage_profiling) {
        std::cout << "Stage profiling info:" << std::endl;
        for (const auto& stage : contract.stages) {
            std::cout << "  " << stage.stage_name << " : " << stage.stage_timing << " ms" << std::endl;
        }
    }

    std::cout << "max_abs_error=" << error << "\n";
    std::cout << "max_relative_error=" << max_rel_error << "\n";
    std::cout << "num_moduli=" << contract.num_moduli << "\n";
    std::cout << "fallback_mode=" << (contract.fallback_mode.empty() ? "None" : contract.fallback_mode) << "\n";
    std::cout << "passed_precision_gate=" << (passed_gate ? 1 : 0) << "\n";
    std::cout << "used_cuda_graph=" << (contract.used_cuda_graph ? "true" : "false") << "\n";
    std::cout << "used_custom_mma=" << (contract.used_custom_mma ? "true" : "false") << "\n";
    std::cout << "scheme_ii_backend=" << (contract.scheme_ii_backend.empty() ? "cublas" : contract.scheme_ii_backend) << "\n";

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));

    if (max_rel_error > 1e-6) {
        return 1;
    }
    return 0;
}
