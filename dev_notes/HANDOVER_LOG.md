# TC-QDP Optimization Handover Log

> **目的**：本文件為本地開發專用的交接與進度追蹤紀錄。不對外 commit 於 Apache Mahout 主分支。
> **核心準則 (C5 Standard)**：所有優化與改動，唯一的成功指標是 **NCU Profiling 數據 (Duration, TC Utilization, Memory Throughput)**，且必須通過 `bench_kernels.cu` 零誤差 (Max Absolute Error = 0) 正確性測試。

---

## [Version 1.0.0] - Stage: Infrastructure & Operator Fusion

**時間/階段**：Phase 1 (基礎建設與小量級優化)
**當前狀態**：已完成 (Completed)

### 🟢 本階段已完成進度 (Completed in v1.0.0)
1. **建立 TDD 基準測試環境 (PR-A 基礎)**：
   - 繞過 Rust 限制，完成純 C++ `CMakeLists.txt` 與 `bench_kernels.cu` 的編譯與測試。
   - 建立 NCU profiling 與 JSON 報表轉換流水線。
2. **算子融合實作 (Operator Fusion, PR-C)**：
   - 實作 `iqp_phase_fwt_normalize_tc_kernel` 於 `iqp_tc.cu`。
   - 針對小 Qubit ($n \le 10$)，成功將 Phase 計算、FWT 與 Normalize 融合在 Shared Memory，消除 Global Memory Round-trip。
   - **成效**：最大絕對誤差為 `0`，效能從 4200us 大幅降低至 27us (約 150 倍提升)。
3. **引入 AdaptiveGEMM 引擎**：
   - `AdaptiveOzaki.h` 與 `AdaptiveOzaki.cu` 接入專案，配置 `-arch=sm_89` (TF32/INT8 Tensor Core 支援) 編譯成功，並於程式碼中預留呼叫位置。

### 🔴 尚未完成與釐清事項 (Clarifications)
- **尚未實作 GEMM 轉換**：雖然 `AdaptiveOzakiEngine` 已引入，但目前**尚未**將大型 Qubit 的 Hadamard 蝴蝶運算結構改寫為 GEMM 矩陣乘法。

---

## [Version 1.1.0] - Stage: FWT to GEMM Conversion (即將進行)

**時間/階段**：Phase 2 (核心演算法置換)
**當前狀態**：計畫中 (Planned)

### 🔵 下一階段行動綱領 (Next Action Items for v1.1.0)
**目標：將大型 Qubit ($n > 10$) 的 FWT 替換為 Tensor Core GEMM 結構**

1. **矩陣重構 (Tensor Reshaping)**：
   - 撰寫記憶體重新排列的 Kernel，將一維狀態向量 ($\vec{v}$) 依照 FWT 的階層需求，Reshape 成可供 `cublas` 或 AdaptiveGEMM 處理的二維矩陣 $B$。
2. **建構變換矩陣 $A$**：
   - 建立並載入各個 Stage 所需的 Hadamard 區塊矩陣 $A$。
3. **引擎點火 (Engine Execution)**：
   - 於 `iqp_tc.cu` 中呼叫 `engine.execute(d_A, d_B, d_C, m, n, k)`。
   - 利用 Ozaki Scheme 自動將 FP64 資料拆解給 INT8 Tensor Core 計算，算出 $C$ 矩陣。
4. **驗證與 Profiling**：
   - 擴展 `bench_kernels.cu` 以測試 $n=12$ 或 $n=14$ 的情況。
   - 利用 NCU 擷取 Tensor Core 啟動率與加速倍率，確保 Max Error = 0。

---
*Note: 每次切換工作或交接時，請於文件最上方新增 `[Version X.X.X] - Stage: ...` 區塊，並詳細記錄改變與下一步。*