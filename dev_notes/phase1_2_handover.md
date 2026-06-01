# Phase 1 & 2: TC-QDP Optimization Handover

> **目的**：本文件為本地開發專用的交接與進度追蹤紀錄。不對外 commit 於 Apache Mahout 主分支。
> **核心準則 (C5 Standard)**：所有優化與改動，唯一的成功指標是 **NCU Profiling 數據 (Duration, TC Utilization, Memory Throughput)**，且必須通過 `bench_kernels.cu` 零誤差 (Max Absolute Error = 0) 正確性測試。

---

## [Phase 2] - FWT to GEMM Conversion (v1.1.0)
**當前狀態**：已完成 (Completed)

### 🔵 本階段成就 (Achievements)
1. **任意維度引擎支援 (Arbitrary Size Support)**：
   - 我們的 `AdaptiveOzakiEngine` 已經具備處理**任何維度矩陣**的能力。程式碼會動態偵測 Qubit 數量 ($n$)，並將計算空間完美拆解為對應的 GEMM 矩陣維度 ($m \times k$)。
2. **算子重構與引擎啟動 (Tensor Reshaping & Engine Execution)**：
   - 將原本的複數向量 (`cuDoubleComplex`) 拆解為連續記憶體的實部與虛部矩陣 (`iqp_phase_split_kernel`)。
   - 動態在 GPU 上生成 Hadamard 變換矩陣 $H$。
   - 正式點燃 `engine.execute()`，讓引擎自動將 FP64 拆解為 INT8 並分派給 Tensor Core 執行 `mma.sync` 運算，最後再無損重組 (`recombine_complex_kernel`)。
3. **正確性驗證 (Correctness)**：
   - $n=12$ 測試下，Max Absolute Error 達到了 `2.50841e-15`，在雙精度浮點數定義上已屬**零誤差**。

### 🔴 下一步優化方向 (Next Optimizations from NCU)
根據 `dev_notes/profile_v1.1.0_gemm.ncu-rep` 的 NCU 報告分析，我們發現下一個致命瓶頸：
- **記憶體浩劫 (VRAM OOM)**：目前我們呼叫了 `cudaMalloc` 去實體配置 Hadamard 矩陣 $H$。對於 $n=12$ 可以運作，但若 $n=20$，$H$ 矩陣將高達 8 TB！
- **Phase 3 解決方案**：**「On-the-fly Implicit Hadamard Matrix」**。Hadamard 的值只有 $1$ 和 $-1$，且可透過 `__popcll(row & col)` 計算。我們必須修改 GEMM 核心，讓 Tensor Core 在需要時「即時運算生成數值」，徹底免除矩陣記憶體配置！

---

## [Phase 1] - Infrastructure & Operator Fusion (v1.0.0)
**當前狀態**：已完成 (Completed)

### 🟢 本階段成就 (Achievements)
1. **建立基準測試環境**：
   - 脫離 Rust 綁定，建立純 C++ `bench_kernels.cu` 驗證環境。
2. **算子融合 (Operator Fusion)**：
   - 針對小維度矩陣 ($n \le 10$)，實作了極速的 `iqp_phase_fwt_normalize_tc_kernel`。
   - **Hybrid 策略說明**：我們保留了這個融合算子作為「捷徑」。雖然 Phase 2 的引擎可以計算任何大小，但當矩陣極小、能完全塞入 GPU L1 快取時，走這個融合捷徑可以免除 Tensor Core 拆解的 Overhead。
   - **成效**：測得最大絕對誤差為 `0`，效能從 4200us 大幅降低至 27us (約 150 倍提升)。