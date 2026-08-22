#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>
#include <cstdint>

namespace ozaki {

struct PreScanStats {
    double dynamic_range;
    double sparsity;
    int effective_rank;
};

enum class ExecutionMode {
    Phase22,
    Phase23Hetero,
    Phase24ExtremeMix,
    Phase25Stacking,
    Phase26HybridOzaki,
    SchemeI
};

struct OzakiConfig {
    double target_epsilon = 1e-10;
    size_t vram_limit_mb = 6144;
    ExecutionMode mode = ExecutionMode::Phase22;
    bool enable_fp64 = false;
    bool enable_fp32 = false; // New for Phase 24
    bool enable_fp16 = false; // New for Phase 24
    bool enable_residual = false;
    bool enable_partial_residual = false;
    bool enable_kernel_fp64 = false;
    int split_fp64_bits = 8;
    int split_fp32_bits = 0;  // New for Phase 24
    int split_fp16_bits = 0;  // New for Phase 24
    int split_tc_bits = 32;
    int split_residual_bits = 13;
    int warp_fp64 = 2;
    int warp_tc = 4;
    int warp_residual = 2;
    double tile_error_guard = 1e-9;
    
    // Scheme I specific fields
    int  scheme_i_mantissa_bits = 34;
    bool scheme_i_triangular = true;
    int  scheme_i_pair_threshold = 4;      // a+b <= 4 -> 15 pairs, 0-based
    bool scheme_i_use_cublas_int8 = true;  // B v0; flip is PR4b
    bool scheme_i_fused = false;           // PR6
};

class AdaptiveOzakiEngine {
public:
    explicit AdaptiveOzakiEngine(const OzakiConfig& config);
    ~AdaptiveOzakiEngine();

    void execute(const double* d_A, const double* d_B, double* d_C, int m, int n, int k);

    // Workspace management
    void allocateWorkspace(int m, int n, int k);
    void freeWorkspace();
    size_t workspaceBytes() const;

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
    
    int ws_m_ = 0, ws_n_ = 0, ws_k_ = 0;
    ExecutionMode ws_mode_ = ExecutionMode::Phase22;
    bool workspace_allocated_ = false;
    cublasHandle_t cublas_handle_;
};
} 
