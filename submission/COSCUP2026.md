# COSCUP 2026 投稿文案

## 難易度定位

Advanced

## 建議講題

【Advanced】突破 FP64 限制：在 RTX 4060 透過 INT8 Tensor Core 實作 Ozaki Scheme 逼近 A100 效能

## Abstract

消費級 NVIDIA Ada GPU 的 FP64 算力長期受限，使科學計算、量化金融與數值線性代數常被迫在「正確性」與「效能」之間做取捨。本議程展示一個手刻 CUDA GEMM kernel：透過 Ozaki Scheme、CRT/RNS 重建與 INT8 Tensor Core 的組合，在 RTX 4060 上將原本受限於 FP64 的矩陣乘法，轉換成可高吞吐執行的低精度路徑，再把結果重建回高精度輸出。報告將說明從硬體限制、演算法拆解、PTX `mma.sync` 實作，到 Shared Memory bank conflict 與 register pressure 的處理方式，並展示 NCU profiling 與效能比較。目標聽眾為 HPC / AI 開發者、CUDA 工程師與需要高精度 GEMM 的應用開發者；先備知識建議具備 C++、GEMM、CUDA 記憶體階層與基本 GPU kernel 調校經驗。

## Notes to Reviewers

30 分鐘演講建議切分如下：

1. 0-5 分鐘：說明消費級 GPU 的 FP64 限制與實際痛點，建立問題背景。
2. 5-12 分鐘：介紹 Ozaki Scheme 與 CRT/RNS 如何把高精度 GEMM 拆成 Tensor Core 可處理的子問題。
3. 12-20 分鐘：展示 PTX `mma.sync`、Tensor Core 資料流、Shared Memory 排佈與 bank conflict / register pressure 的處理。
4. 20-26 分鐘：展示效能數據、NCU profiling、與 cuBLAS FP64 的對照結果。
5. 26-30 分鐘：說明函式庫化方向與應用場景，包括 MATLAB/C++ 使用模式，以及透過 pybind11 或 PyTorch C++ Extension 進入 Python 生態系的路線圖。