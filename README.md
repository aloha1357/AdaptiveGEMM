# AdaptiveGEMM: RTX 4060 上的 INT8 Tensor Core Ozaki Scheme FP64 加速

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.18%2B-064F8C.svg)](https://cmake.org/)
[![CI](https://img.shields.io/badge/GitHub_Actions-enabled-2088FF.svg)](.github/workflows/ci.yml)

AdaptiveGEMM 是一個面向 NVIDIA Ada 架構消費級 GPU 的 CUDA C++ GEMM 專案，核心目標是在 FP64 算力受限的 RTX 4060 / 4090 等卡上，透過 INT8 Tensor Core、Ozaki Scheme 與 CRT/RNS 重建，把高精度矩陣乘法推進到接近資料中心級 GPU 的效能區間。

這個 `public_release/` 目錄是可直接搬到 GitHub 的獨立 repo 視圖：包含授權、原始碼、示範程式、測試、GitHub Actions 與投稿摘要，不再依賴父層工程。

Note: the GitHub Actions workflow includes a GPU validation job that runs on a self-hosted Linux runner with the `gpu` label. Make sure that runner is enabled in the GitHub repository settings before relying on the action.

Important: this repository is open source under Apache 2.0. If you want separate commercial licensing, keep it outside the public license.

## Motivation

科學計算、量化金融、數值線性代數與物理模擬，長期依賴 FP64 GEMM 來維持數值穩定性。但 NVIDIA 消費級 GPU 的 FP64 throughput 往往遠低於其 Tensor Core 峰值，導致一個很現實的落差：

- 演算法需要 double precision。
- GPU 硬體最擅長的卻是低精度 Tensor Core。
- 傳統 cuBLAS FP64 在消費級卡上常常被硬體規格直接卡死。

AdaptiveGEMM 的切入點，就是把「高精度需求」拆成「可在 Tensor Core 上高吞吐執行的低精度子問題」，再透過 Ozaki Scheme 與重建流程把結果拉回 FP64 可接受的精度區間。

## Technical Highlights

### 1. Algorithm: Ozaki Scheme for precision reduction

本專案採用 Ozaki Scheme 的核心精神，把 FP64 輸入切分成多個可控誤差的片段，再把部分乘積轉換成更容易被 Tensor Core 吞吐的型態。對外看起來是 GEMM，對內實際上是：

1. 先做 range analysis 與 scaling。
2. 把高精度數值映射到 INT8 / low-precision 片段。
3. 透過 CRT / RNS 類型的重建與補償，把累積誤差壓回可控範圍。

這種做法的本質不是「假裝有 FP64」，而是「用硬體最強的路徑做出可驗證的高精度結果」。

### 2. Hardware path: PTX `mma.sync` drives Tensor Core

底層 kernel 的重點不是一般 CUDA FMA，而是直接把運算導向 Tensor Core pipeline。也就是：

- 以 PTX `mma.sync` 觸發矩陣乘加。
- 讓 warp-level tile feed Tensor Core。
- 把資料佈局對齊到 Tensor Core 期望的 fragment / operand 排列。

這樣做的目的很直接：把原本 FP64 bottleneck 轉成 high-throughput INT8 Tensor Core bottleneck，然後再由重建層把精度拉回來。

### 3. Memory optimization: bank conflict and register pressure control
<!-- Insert Nsight Compute screenshots below -->
### Speed of Light

<p align="center">
  <img src="docs/profile_sol.png" alt="Speed of Light" width="88%" />
</p>

### Pipe Utilization

<p align="center">
  <img src="docs/profile_pipe.png" alt="Pipe Utilization" width="88%" />
</p>

如果 Tensor Core 是算力核心，Shared Memory 與 register file 就是這個專案真正的戰場。公開版 README 建議你在此保留以下說明：

- Shared Memory bank conflict 的排除策略。
- warp skew / sync hazard 的修正方法。
- register pressure 上限與 occupancy 之間的取捨。
- tile 與 accumulator 的雙階段配置方式。

這些細節不只是 performance tuning，而是能否穩定跑出 benchmark 的關鍵。

## Performance & Profiling

> 這一節保留給你貼 NCU 圖、throughput 數據、以及和 cuBLAS FP64 的倍數比較。

### Benchmark placeholder

- GPU model: `RTX 4060`
- Architecture: `Ada / sm_89`
- Matrix size: `2048 x 2048` or your target size
- Peak metric: `GFLOPS` / `TFLOPS`
- Accuracy metric: `max error`, `avg error`, `relative error`

### Nsight Compute placeholder

請在此貼上：

- Roofline / Compute Throughput 截圖
- Tensor Core utilization
- Shared Memory efficiency
- Register usage / occupancy
- Memory throughput and stall reason

### Comparison placeholder

| Method | Precision | Time | Throughput | Notes |
|---|---:|---:|---:|---|
| cuBLAS FP64 | FP64 | TODO | TODO | baseline |
| AdaptiveGEMM | FP64-like | TODO | TODO | INT8 Tensor Core + Ozaki |

## Use Cases & API

AdaptiveOzaki 不只是 benchmark，它應該被使用成一個函式庫。

### Use cases

- 科學計算：PDE、線性方程組、Krylov method、矩陣分解。
- 量化金融：risk simulation、Monte Carlo accelerator、large-scale covariance GEMM。
- 工程計算：任何需要 double precision，但又想吃 Tensor Core 吞吐的場景。

### API direction

目前的 C++ API 會以簡單、可嵌入的形式暴露：

- `AdaptiveOzakiEngine`
- `OzakiConfig`
- `execute(...)`

未來路線圖：

- `pybind11`：把核心 GEMM 包成 Python 可直接呼叫的 module。
- PyTorch C++ Extension：讓研究用模型能直接接上。
- MATLAB / C++ integration：把這個 kernel 當成可重用的數值後端。

### License and commercial terms

This repository is released under Apache 2.0.

Apache 2.0 already allows academic, personal, and commercial use. If you want to offer an additional paid commercial license, proprietary redistribution terms, support, or an exclusive deployment agreement, that is handled separately as a dual-license or commercial-license track.

## Quick Start

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Run demo

```bash
./build/Release/adaptive_gemm_demo
```

For Windows with Visual Studio generators, open the root directory (`.`) as the build root in VS Code or use the generated `build` directory directly.

### Run precision test

```bash
ctest --test-dir build -C Release --output-on-failure
```

## Repository Layout

```text
.
├── .github/
│   └── workflows/
├── include/
├── src/
├── tests/
├── CMakeLists.txt
├── LICENSE
├── NOTICE
├── README.md
├── examples/
│   └── demo_main.cpp
└── submission/
    └── COSCUP2026.md
```

## Notes for Open Source Release

This repository view is intentionally narrow.

- It keeps the technical narrative focused on GEMM, not on every experimental phase.
- It preserves the exact story reviewers need: pain point, method, profiling evidence, and application path.
- It leaves space for you to replace placeholder metrics with final NCU evidence before publishing.