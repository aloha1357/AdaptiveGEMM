# Phase 3 & 4 Handover: Matrix-Free Tensor Core Engine & Fusion Limits

## 1. 核心實作成果 (Phase 3 Global Optimum)
在 Phase 3 中，我們成功打造了超越 PyTorch 原生限制的量子模擬底層引擎，主要包含三大核心突破：

1. **智慧分流架構 (Adaptive Routing)**
   - 修改了 `kernel_config.h`，將 `FWT_SHARED_MEM_THRESHOLD` 設為 12，並動態配置 `cudaFuncAttributeMaxDynamicSharedMemorySize` 至 64KB。
   - 當 $N \le 12$ 時，資料完全不出 SM (Streaming Multiprocessor)，利用 Shared Memory FWT 達成 **0.41 ms** 的極速，比 PyTorch (24.45 ms) **快 60 倍**。

2. **免矩陣即時生成 (Matrix-Free On-the-fly Generation)**
   - 徹底消滅了 Hadamard 矩陣的 Global Memory 需求。利用位元運算 `__popcll(row & col) & 1` 結合硬體 Warp-level 暫存器，在 GPU 核心內部即時生成矩陣元素。
   - 成功突破 $N=16$ 的 VRAM 物理限制（PyTorch 需要 34.3GB 直接 OOM，而我們只需極少量記憶體即可在 5.4 秒內完成計算）。

3. **Ozaki INT8 降維打擊 (FP64 Precision via Tensor Cores)**
   - RTX 4090 的 FP64 算力受到嚴重人工限制 (~1.3 TFLOPs)，若使用純 FP64 計算會極度緩慢。
   - 我們導入了 7 個質數的中國剩餘定理 (CRT)，將 FP64 模擬拆解為 INT8 矩陣乘法，成功解鎖 RTX 4090 高達 660 TOPS 的 Tensor Core 算力。雖然 ALU 還原運算帶來了約 1300 ms 的 Fixed Overhead，但成功以運算時間換取了無限的記憶體空間。

## 2. 效能基準測試對決 (RTX 4090, Batch=128)

| Qubits ($N$) | PyTorch Eager | QDP Tensor Core Engine | 狀態解析 |
| :--- | :--- | :--- | :--- |
| **10** | 1.29 ms | **0.10 ms** | FWT Shared Memory 絕對優勢 |
| **12** | 24.45 ms | **0.41 ms** | FWT 物理極限，快 60 倍 |
| **14** | 339.80 ms | **325.58 ms** | Matrix-Free Tensor Core 反超 |
| **15** | **1306.50 ms** | 1390.72 ms | PyTorch 爆發最後的 VRAM 頻寬；QDP 承擔 ALU 還原代價 |
| **16** | 💀 **OOM** | **5460.88 ms** | PyTorch 崩潰；QDP 絕對制霸 |

## 3. 失敗的嘗試：為什麼放棄 Phase 4 終極算子融合？
在開發過程中，我們嘗試了「Phase 4 Ultimate Fusion」，意圖將 Phase 計算、質數取模 (Modulo) 與 Tensor Core MMA 融合成單一 Kernel，以省去中間 32MB 的 Global Memory 讀寫。
結果 **效能不增反減 (從 300 ms 退步到 360 ms)**。原因已透過 Nsight Compute (NCU) 證實：
1. **L2 快取崩潰 (Cache Thrashing)**：融合後，GPU 必須同時將實部與虛部展開成高達 224MB 的巨型 INT8 陣列，直接撐爆 RTX 4090 的 72MB L2 Cache。
2. **ALU 瓶頸惡化**：為了避開快取問題，若將實部/虛部分兩趟計算，則極度昂貴的 `sincos` (三角函數) 必須被重複計算兩次，大量消耗了指令週期。

**結論**：目前的 Phase 3 (將 Phase Split 獨立，再進行 2-Pass MMA) 是在運算力、L2 快取容量與暫存器壓力之間，所能達到的**單卡物理全局最佳解**。

## 4. 微架構優化 (Micro-architecture Optimizations)
感謝開源社群 (PR20260601) 的建議，我們針對 Phase Encoding 算子進行了極致的底層優化：
1. **消滅 Warp Divergence**：將原本的 `if ((x >> i) & 1U)` 控制流，替換為無分支的 FMA (Fused Multiply-Add) 數學運算 `(double)((x >> i) & 1U)`。確保同一個 Warp 內的 32 個 Threads 能在同一個時脈週期內全速完工，徹底消除了執行緒凍結的懲罰。
2. **Zero-Overhead 常數傳遞 (Constant Memory)**：捨棄了在 Kernel 內使用 `pow` (SFU) 計算 `1/√2^n` 的作法。改為在 CPU Host 端預先計算好 `norm_factor`，並以純量參數傳遞給 Kernel。這讓編譯器能將其放入 GPU 的 Constant Memory / Registers，達成了零負擔且零精度損失的完美正規化。

## 5. 基準測試異常分析：Windows WDDM 記憶體分頁現象
在 Python 執行基準測試時，我們觀察到了極為驚人的效能異常：
- 當 Python 迴圈先執行 `PyTorch Eager` 再呼叫外部 `C++ QDP Engine` 時，C++ 引擎在 $N=15$ 的時間會從正常的 **1.4 秒暴增到 45 秒**！
- **真因分析**：PyTorch 在 $N=15$ 時會向作業系統請求並佔用高達 **8.5 GB** 的連續 VRAM。即使呼叫了 `torch.cuda.empty_cache()`，Windows WDDM (Windows Display Driver Model) 由於 VRAM 碎片化，會強制將新生成的 C++ GPU 行程的記憶體**分頁 (Paging) 到系統的 DDR5 記憶體上 (Shared GPU Memory)**。
- 透過 PCIe 匯流排讀取系統記憶體的頻寬極低，導致原本 1.4 秒的計算被硬生生拖慢到了 45 秒。這完美證明了在 GPU 開發中，**記憶體駐留狀態與 OS 的分頁機制對效能的影響是毀滅性的**。而在乾淨的 GPU 狀態下（例如單獨執行 C++，或 $N=16$ PyTorch 直接 OOM 未佔用記憶體時），QDP 引擎的真實效能依然穩如泰山 (1.4s / 5.4s)。

## 6. 下一步：Stage 5 (Multi-GPU / 分散式計算) & TC-FWT
單張 GPU 的極限已被完全榨乾。針對 $N \ge 30$ 的真實量子霸權規模，下一步的雙軌策略：
1. **演算法突破 (TC-FWT)**：嘗試將 $O(N \log N)$ 的 FWT 演算法映射到 INT8 Tensor Core，結合兩者的終極優勢。
2. **硬體拓展 (NCCL)**：引入 NCCL 或 CUDA IPC，將巨型狀態向量切割並跨 PCIe/NVLink 分佈到多張顯示卡進行平行運算。
