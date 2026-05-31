# Phase 24 & 26 Performance Tuning and Occupancy Optimization Report

## 1. Context & Goal (專案背景與當前目標)
*   **目前處於哪個研發階段？** Phase 24 (ExtremeMix) 與 Phase 26 (HybridOzaki) 的深度效能調優階段。
*   **最主要的目標是什麼？** 
    本次最佳化完全依據 **Nsight Compute (ncu)** 的 Profiling 數據進行。主要為了解決兩大瓶頸：
    1.  `precompute_modulo_kernel` 在前次改版後因記憶體非連續寫入 (Uncoalesced Global Memory Writes) 導致耗時暴增至 50ms。
    2.  `decoupled_crt_pass_kernel` 面臨極端的 Register Pressure (每 Thread > 128 Registers)，導致 SM 理論佔用率 (Theoretical Occupancy) 被卡死在 33.33%，無法有效隱藏記憶體延遲。
*   **重點指標 (Metrics)**：SM Achieved Occupancy、Memory Throughput、Kernel Duration。

## 2. Architecture & Design (架構與設計狀態)
本次基於 Nsight Profiling 數據進行了以下四個核心架構級別的優化：

1.  **Block-Tiled Memory Layout (連續記憶體拼貼)**:
    將 A8 餘數矩陣的儲存佈局改為 `128x32`，B8 改為 `64x32`。這確保了 Tensor Core 載入資料所使用的 `cp_async_16` 指令在 Warp 級別能夠達成 100% 的 Coalesced Memory Access，使得 DRAM 讀取區塊數減少近一半。
2.  **Coalesced Precompute Kernel (佈局轉換最佳化)**:
    為了配合上述的 Tiled Layout，將 Precompute 階段的執行區塊由原本的 `16x16` 改寫為 `32x32` block。透過精準讓 `threadIdx.x` 直接對齊 Layout 內部連續的 `local_k` 維度，徹底消除了原本 Uncoalesced Write 帶來的巨大延遲。
3.  **Variable Life-cycle Analysis & Register Spilling (佔用率突破)**:
    針對 CRT 核心的 Register Pressure 問題，我們在 Kernel 加上 `__launch_bounds__(256, 3)` 強制限制編譯器。藉由分析變數生命週期，我們讓編譯器將非內部高頻迴圈 (Critical Inner Loop) 使用的 64-register 陣列 (`final_weighted_sum`) 溢出 (Spill) 存放至高速的 L1 Cache 中。這項改動成功犧牲極少量的 L1 頻寬，換取了 SM 佔用率成功提升至 **50%~60%** (3 Blocks / SM)，大幅改善效能。
4.  **Sliding Pointers in Inner Loop (消除 ALU 瓶頸)**:
    移除了 CRT 內部迴圈對於 Index 的除法與取模 (Modulo) 運算，改以常數偏移量的 Sliding Pointer 推進，降低了 ALU 的運算壓力。

## 3. Setup & Environment (環境與建置指南)
*   **建置專案 (Windows MSVC + NVCC)**:
    ```cmd
    cd public_release
    cmake --build build --config Release
    ```
*   **執行效能測試**:
    ```cmd
    .\build\Release\adaptive_gemm_demo.exe
    ```
*   **執行 Nsight Compute Profiling (驗證效能依據)**:
    ```cmd
    ncu --print-summary per-kernel -f .\build\Release\adaptive_gemm_demo.exe
    ```

## 4. Current Status (最新進度與已知問題)
本次修改已透過 `ncu` 進行完整數據比對與驗證：

*   **Precompute 效能跳躍**：透過 Coalesced 調整，耗時從異常的 ~50ms 降回預期的 **~9ms** 以內。
*   **Phase 24 (ExtremeMix) 效能數據**:
    *   **矩陣尺寸**: 2048 x 2048
    *   **Average time**: ~128 ms 至 130 ms
    *   **Throughput**: ~133 GFLOPS
    *   **Max error vs cuBLAS FP64**: 3.55271e-15 (維持完美的 FP64 數值精度)
*   **Phase 26 (HybridOzaki) 效能數據**:
    *   **Average time**: ~81 ms 至 82 ms
    *   **Throughput**: ~211 GFLOPS
    *   **Max error vs cuBLAS FP64**: 0.221941 (混合精度表現符合預期)
*   **cuBLAS FP64 整合評估 (已捨棄)**：
    透過 Nsight Compute 分析我們將高頻區域委派給 `cublasDgemm` 時發現，受限於 RTX 4060 (Ada Lovelace) 上極端閹割的 FP64 單元 (1/64 比例)，呼叫 cuBLAS 的耗時反而遠大於我們自己撰寫的 Naive Masked 混合 Kernel，因此最終策略將其捨棄，證明在特定閾值分割下自行實作是最佳解。
*   **已知問題**：根據 `ncu` 報告，目前的 `add_residual_kernel` DRAM Throughput 高達 90% 以上，已經成為純粹的 Memory Bound Kernel。

## 5. Next Steps (給接手者的下一步行動建議)
1.  **提交變更 (Commit Changes)**：
    目前的最佳化結果均存在於本地端，請整理 `AdaptiveOzaki.cu` 與 `demo_main.cpp` 的變更，提交 Commit 並推播回 Git Repository。
2.  **Kernel Fusion 評估**：
    針對 `ncu` 測出 `add_residual_kernel` Memory Bound 的問題，評估是否能將最後的矩陣加法 (Add Residual) 直接融合 (Fuse) 到 `decoupled_crt_pass_kernel` 寫回 Global Memory 的階段，以節省一次矩陣讀寫的巨大頻寬消耗。
3.  **不同架構的 Register 策略測試**：
    目前 `__launch_bounds__(256, 3)` 與 L1 Spilling 策略是在 RTX 4060 (sm_89) 上最佳化的結果。若未來要部署於 Hopper (H100) 或 Ampere (A100) 等 Register File 較大的硬體，建議重新執行 Profiling，或許可以解除限制並達到更高的吞吐量。