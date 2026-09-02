#include "scheme_ii_int192.h"
#include "DoubleDouble.h"
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <string>

namespace ozaki {

struct PreScanStats {
    double dynamic_range;
    double sparsity;
    int effective_rank;
};

enum class ExecutionMode {
    Phase22,
    Phase22_Fused,
    Phase23Hetero,
    Phase24ExtremeMix,
    Phase25Stacking,
    Phase26HybridOzaki,
    SchemeI,
    SchemeII,
    SchemeII_Fused
};

enum class PrecisionPolicy {
    Fixed,
    Auto,
    NativeFallback
};

enum class SchemeIIGemmBackend {
    CuBLAS,
    CustomMMA
};

struct PipelineStage {
    std::string stage_name;
    double stage_timing = 0.0;
    size_t workspace_size = 0;
};

struct SchemePlan {
    int num_split = 0;
    int bits_per_int8 = 0;
    int num_moduli = 0;
    std::string pair_schedule = "";
};

struct PrecisionContract {
    std::string mode;
    std::string input_type = "FP64";
    int target_mantissa_bits = 53;
    int num_split = 0;
    int bits_per_int8 = 0;
    int num_moduli = 0;
    int crt_bits = 0;
    int required_crt_bits = 0;
    bool residual_enabled = false;
    bool residual_used = false;
    bool native_fp64_used = false;
    std::string fallback_mode = "";
    std::string failure_reason = "";
    double max_relative_error = -1.0;
    bool passed_precision_gate = false;
    bool used_cuda_graph = false;
    bool used_true_fused_kernel = false;
    bool used_scheme_ii_fused = false;
    bool used_custom_mma = false;
    std::string scheme_ii_backend = "";
    int scheme_ii_group_size = 0;
    int scheme_ii_gemm_launches = 0;
    bool used_partial_crt_epilogue = false;
    int scheme_ii_mma_variant = 0;
    std::string scheme_ii_prologue_fusion = "none";
    std::vector<PipelineStage> stages;
};

struct GEMMInput {
    int M;
    int N;
    int K;
    std::string element_type = "float64";
    std::string op_A = "N";
    std::string op_B = "N";
    double alpha = 1.0;
    double beta = 0.0;
};

struct OzakiConfig {
    int target_mantissa_bits = 53;
    PrecisionPolicy precision_policy = PrecisionPolicy::Fixed;
    bool report_precision_contract = true;
    bool allow_native_fp64_fallback = false;

    // Scheme I specific fields
    int scheme_i_num_split = 0;
    int scheme_i_bits_per_int8 = 0;
    int scheme_i_pair_threshold = -1;

    // Scheme II specific fields
    int scheme_ii_num_moduli = 15;
    int scheme_ii_num_split = 0;
    int scheme_ii_bits_per_int8 = 0;
    bool scheme_ii_fastmode = true;
    bool scheme_ii_reuse_workspace = true;
    bool scheme_ii_use_batched_gemm = true;
    bool scheme_ii_use_cuda_graph = false;
    // --use_true_fused_kernel with group_size < num_moduli selects M8.3
    // partial CRT (used_partial_crt_epilogue=true, used_true_fused_kernel=false).
    // group_size >= num_moduli selects M8.4 all-moduli true fused.
    bool scheme_ii_use_true_fused_kernel = false;
    SchemeIIGemmBackend scheme_ii_gemm_backend = SchemeIIGemmBackend::CuBLAS;
    int scheme_ii_moduli_group_size = 4;
    int scheme_ii_mma_variant = 0;
    int scheme_ii_prologue_fusion = 0; // 0=none, 1=tile_local
    bool enable_stage_profiling = false;
    std::string scheme_ii_ablate = "";

    // Legacy fields (kept for backward compatibility, specifically Phase22)
    double target_epsilon = 1e-10;
    size_t vram_limit_mb = 6144;
    ExecutionMode mode = ExecutionMode::Phase22;
    bool enable_fp64 = false;
    bool enable_fp32 = false;
    bool enable_fp16 = false;
    bool enable_residual = false;
    bool enable_partial_residual = false;
    bool enable_kernel_fp64 = false;
    int split_fp64_bits = 8;
    int split_fp32_bits = 0;
    int split_fp16_bits = 0;
    int split_tc_bits = 32;
    int split_residual_bits = 13;
    int warp_fp64 = 2;
    int warp_tc = 4;
    int warp_residual = 2;
    double tile_error_guard = 1e-9;
    int scheme_i_mantissa_bits = 34;
    bool scheme_i_triangular = true;
    bool scheme_i_use_cublas_int8 = true;
    bool scheme_i_fused = false;
    bool enable_skip_scalA = false;
};

class AdaptiveOzakiEngine {
public:
    explicit AdaptiveOzakiEngine(const OzakiConfig& config);
    ~AdaptiveOzakiEngine();

    void execute(const double* d_A, const double* d_B, double* d_C, int m, int n, int k);
    void execute(const double* d_A, const double* d_B, double* d_C, const GEMMInput& input);
    void execute(const DoubleDouble* d_A, const DoubleDouble* d_B, DoubleDouble* d_C, int m, int n, int k);
    void execute(const DoubleDouble* d_A, const DoubleDouble* d_B, DoubleDouble* d_C, const GEMMInput& input);

    // Workspace management
    void allocateWorkspace(int m, int n, int k);
    void freeWorkspace();
    size_t workspaceBytes() const;

    const SchemePlan& getSchemePlan() const { return current_plan_; }
    const PrecisionContract& getPrecisionContract() const { return current_contract_; }
    PrecisionContract& mutablePrecisionContract() { return current_contract_; }

    static void bitShiftToINT8(const double* d_src, int8_t* d_dst_int8, int elem_count, int exponent_shift);
    static void computeResidualGradedRing(const double* d_A, const double* d_B, const double* d_C, double* d_R, int m, int n, int k);
    static void accumulateSlicedProduct(const int8_t* d_A_int8, const int8_t* d_B_int8, double* d_C, int m, int n, int k, int total_shift);
    static void accumulateFusedSlicedProductWMMA(const double* d_A, const double* d_B, double* d_C, int m, int n, int k, int shift_A, int shift_B);

protected:
    PreScanStats analyzeMatrix(const double* d_A, const double* d_B, int m, int n, int k);
    int calculateOptimalTile(int m, int n, int k);

private:
    void profileHardwareRatio();

    OzakiConfig config_;
    SchemePlan current_plan_;
    PrecisionContract current_contract_;
    bool scalA_cached_ = false;
    double ratio_fp64_tc = -1.0; // -1 means not profiled yet
    double ratio_fp32_tc = -1.0;
    double ratio_fp16_tc = -1.0;
    double ratio_tf32_tc = -1.0;
    double ratio_int32_tc = -1.0;

    // Workspace pointers
    int8_t *dA8_h = nullptr, *dA8_l = nullptr, *dB8_h = nullptr, *dB8_l = nullptr;
    uint64_t *dmA_h = nullptr, *dmA_l = nullptr, *dmB_h = nullptr, *dmB_l = nullptr;
    int* d_global_work_queue = nullptr;
    
    // Low-precision buffers for cross-terms
    float *dA_hi_f32 = nullptr, *dA_low_f32 = nullptr;
    float *dB_hi_f32 = nullptr, *dB_low_f32 = nullptr;
    double *dA_hi = nullptr, *dA_low = nullptr;
    double *dB_hi = nullptr, *dB_low = nullptr;
    
    // Scheme I workspace
    int8_t* d_A_slices = nullptr;
    int8_t* d_B_slices = nullptr;
    double* d_maxA = nullptr;
    double* d_maxB = nullptr;
    int32_t* d_eA = nullptr;
    int32_t* d_eB = nullptr;
    int32_t* d_row_sum = nullptr;
    int32_t* d_col_sum = nullptr;
    int32_t* d_C_i32 = nullptr;
    
public:
    // Scheme II workspace
    int* d_scheme_ii_eA = nullptr;
    int* d_scheme_ii_eB = nullptr;
    int8_t* d_scheme_ii_A_mod = nullptr;
    int8_t* d_scheme_ii_B_mod = nullptr;
    // N-major (K-contiguous) copy of B_mod for custom grouped MMA Path A.
    int8_t* d_scheme_ii_B_mod_nk = nullptr;
    int32_t* d_scheme_ii_C_mod = nullptr;
    int32_t* d_scheme_ii_primes = nullptr;
    int32_t* d_scheme_ii_inv_table = nullptr;
    Int192* d_scheme_ii_P = nullptr;
    int d_scheme_ii_ws_num_moduli = 0;
    cudaStream_t d_scheme_ii_stream = nullptr;
    cudaStream_t d_scheme_ii_stream2 = nullptr;
    cudaEvent_t d_scheme_ii_ev_b = nullptr;
    cudaGraph_t d_scheme_ii_graph = nullptr;
    cudaGraphExec_t d_scheme_ii_graph_exec = nullptr;
    bool d_scheme_ii_graph_valid = false;
    int d_scheme_ii_graph_m = 0;
    int d_scheme_ii_graph_n = 0;
    int d_scheme_ii_graph_k = 0;
    int d_scheme_ii_graph_num_moduli = 0;
    const double* d_scheme_ii_graph_A = nullptr;
    const double* d_scheme_ii_graph_B = nullptr;
    double* d_scheme_ii_graph_C = nullptr;
    bool d_scheme_ii_graph_custom_mma = false;
private:
    int ws_m_ = 0, ws_n_ = 0, ws_k_ = 0;
    ExecutionMode ws_mode_ = ExecutionMode::Phase22;
    bool workspace_allocated_ = false;
    cublasHandle_t cublas_handle_;
};
} 

