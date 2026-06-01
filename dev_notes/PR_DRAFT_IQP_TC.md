# Pull Request: High-Performance Matrix-Free IQP Encoding via INT8 Tensor Cores

## 🚀 Motivation
The current implementation of IQP (Instantaneous Quantum Polynomial) encoding relies on PyTorch's dense matrix multiplication (`cuBLAS`) for generating state vectors. While this is fast for very small qubit counts ($N \le 12$), it fundamentally hits a "Memory Wall." 

For example, at $N=16$, the dense Hadamard matrix requires **34.3 GB** of continuous VRAM (FP64). This causes immediate Out-Of-Memory (OOM) crashes on consumer GPUs (like the RTX 4090 with 24GB VRAM) and even limits enterprise hardware. We needed a solution that escapes the memory bound and relies purely on computational power.

## 🛠️ Implementation Details
This PR introduces a unified, adaptive GPU engine that completely resolves the memory bottleneck and maximizes hardware utilization:

1. **Adaptive Routing & Shared Memory Unleashed ($N \le 12$)**
   - We dynamically allocate up to 64KB of Shared Memory using `cudaFuncSetAttribute` for smaller scales.
   - For $N \le 12$, the data stays entirely within the Streaming Multiprocessor (SM). We utilize a lightning-fast Shared Memory Fast Walsh-Hadamard Transform (FWT).
   - **Result:** ~60x speedup over PyTorch.

2. **Matrix-Free On-the-fly Generation ($N \ge 14$)**
   - We eliminated the need to allocate the global Hadamard matrix. Elements are generated directly inside the GPU registers on-the-fly using bitwise parity: `__popcll(row & col) & 1`.
   - **Result:** Memory requirement drops from gigabytes to megabytes, unlocking $N=16$ and beyond.

3. **Ozaki INT8 Tensor Core Engine (FP64 Precision)**
   - Consumer GPUs (like Ada Lovelace/RTX 4090) heavily throttle native FP64 calculations (~1.3 TFLOPs).
   - We implemented the **Ozaki scheme combined with the Chinese Remainder Theorem (CRT)** using 7 primes. This allows us to slice the FP64 simulation into multiple INT8 blocks.
   - We feed these blocks into the 660 TOPS INT8 Tensor Cores. Despite the ALU overhead required to reconstruct the FP64 output via CRT, the massive compute throughput allows us to trade compute time for infinite memory scalability.

## 📊 Benchmarks (RTX 4090, Batch Size = 128)

| Qubits ($N$) | PyTorch Eager | QDP Tensor Core Engine | Note |
| :--- | :--- | :--- | :--- |
| **10** | 1.29 ms | **0.10 ms** | 12x Faster (FWT Shared Memory) |
| **12** | 24.45 ms | **0.41 ms** | **60x Faster** (FWT Shared Memory) |
| **14** | 339.80 ms | **325.58 ms** | Matrix-Free Tensor Core takes over |
| **15** | **1306.50 ms** | 1390.72 ms | PyTorch uses 8GB VRAM; QDP uses ~0GB but pays ALU overhead |
| **16** | 💀 **OOM** | **5460.88 ms** | PyTorch crashes; QDP dominates |

*(Note: `torch.compile` fails completely in native Windows environments due to missing Triton support, making this custom C++/CUDA kernel essential for cross-platform stability.)*

## 🔮 Next Steps
- Implement Stage 5: Multi-GPU and NCCL distributed state vector encoding for $N \ge 30$.
- Investigate mapping the $O(N \log N)$ FWT algorithm directly onto Tensor Cores to reduce the algorithmic complexity of the current $O(N^2)$ Ozaki GEMM approach.
