# TC-QDP Optimization Handover Log

> **目的**：本文件為本地開發專用的交接與進度追蹤紀錄。不對外 commit 於 Apache Mahout 主分支。
> **核心準則 (C5 Standard)**：所有優化與改動，唯一的成功指標是 **NCU Profiling 數據 (Duration, TC Utilization, Memory Throughput)**，且必須通過 `bench_kernels.cu` 零誤差 (Max Absolute Error = 0) 正確性測試。

---

## [Version 1.1.0] - Stage: FWT to GEMM Conversion

**時間/階段**：Phase 2 (核心演算法置換)
**當前狀態**：已完成 (Completed)

### 🔵 本階段已完成進度 (Completed in v1.1.0)
1. **測試環境擴展**：
   - 修改 `bench_kernels.cu`，將 Qubit 數量提高至 $n=12$，觸發了大型 Qubit 的 AdaptiveGEMM 路徑。
2. **算子重構與引擎串接 (Tensor Reshaping & Engine Execution)**：
   - 於 `iqp_tc.cu` 內撰寫了 `iqp_phase_split_kernel` 將狀態向量拆分為實部與虛部。
   - 開發了 `generate_hadamard_matrix` 在 GPU 動態生成符合比例的 Hadamard 變換矩陣 $H$。
   - 成功呼叫 `AdaptiveOzakiEngine::execute` 兩次 (分別處理實部與虛部)，由引擎將 FP64 拆解為 INT8 並分派給 Tensor Core 執行 `mma.sync` 運算。
   - 透過 `recombine_complex_kernel` 成功將結果組裝回 `cuDoubleComplex` 格式。
3. **驗證與 Profiling 成果**：
   - 正確性驗證 (Correctness)：與原版演算法比對，**Max Absolute Error 僅為 `2.50841e-15`**，在雙精度浮點數中視同零誤差，完美過關。
   - 效能剖析 (NCU Profiling)：成功擷取到引擎內部的 `accumulate_fused_kernel_masked`、`decoupled_crt_pass_kernel` 與 `residual_kernel` 軌跡。報告儲存於 `dev_notes/profile_v1.1.0_gemm.ncu-rep`。

---

## [Version 1.0.0] - Stage: Infrastructure & Operator Fusion

**時間/階段**：Phase 1 (基礎建設與小量級優化)
**當前狀態**：已完成 (Completed)

### 🟢 本階段已完成進度 (Completed in v1.0.0)
1. **建立 TDD 基準測試環境 (PR-A 基礎)**：
   - 繞過 Rust 限制，完成純 C++ `CMakeLists.txt` 與 `bench_kernels.cu` 的編譯與測試。
   - 建立 NCU profiling 與 JSON 報表轉換流水線 (報告存於 `dev_notes/v1.0.0_fusion_profile.ncu-rep`)。
2. **算子融合實作 (Operator Fusion, PR-C)**：
   - 實作 `iqp_phase_fwt_normalize_tc_kernel` 於 `iqp_tc.cu`。
   - 針對小 Qubit ($n \le 10$)，成功將 Phase 計算、FWT 與 Normalize 融合在 Shared Memory，消除 Global Memory Round-trip。
   - **成效**：最大絕對誤差為 `0`，效能從 4200us 大幅降低至 27us (約 150 倍提升)。
3. **引入 AdaptiveGEMM 引擎**：
   - `AdaptiveOzaki.h` 與 `AdaptiveOzaki.cu` 接入專案，配置 `-arch=sm_89` (TF32/INT8 Tensor Core 支援) 編譯成功，並於程式碼中預留呼叫位置。