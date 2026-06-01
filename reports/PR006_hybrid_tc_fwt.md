# Pull Request: [Stage 5] TDD-6 Hybrid Architecture & Blocked TC-FWT Breakthrough

## 1. What Changed
This PR fundamentally reconstructs the Fast Walsh-Hadamard Transform (FWHT) logic for Quantum State Preparation (IQP Encoding) by implementing the **TDD-6 Hybrid Architecture**. 

Specifically, we:
1. **Removed Warp Divergence**: Replaced `if ((x >> i) & 1U)` with a branchless FMA operation `(double)((x >> i) & 1U)` in phase generation.
2. **Zero-Overhead Constant Memory**: Eliminated the in-kernel `pow` (SFU) calculation for normalization, replacing it with a pre-calculated CPU scalar passed directly to GPU Constant Registers.
3. **Implemented Hybrid Routing**: 
   - $N \le 12$ (Micro-blocks): Routed strictly to Pure SIMT Warp Shuffles and Shared Memory.
   - $N \ge 14$ (Macro-blocks): Routed to the INT8 Tensor Core Ozaki Engine.
4. **Algorithmic Complexity Reduction (Blocked TC-FWT)**: For macro-blocks, we abandoned the naive $O(4^N)$ Massive GEMM. Instead, we implemented a Kronecker Product Decomposition, splitting $N$ (e.g., 14) into $n_1=7$ and $n_2=7$. The operation is now two smaller GEMMs with a memory transpose: $Y = ( (X \times H_{n2})^T \times H_{n1} )^T$. This reduces theoretical complexity from $O(4^N)$ to $O(N \cdot 2^N)$.

## 2. Why
Our microkernel benchmarks (TDD-1 vs TDD-5) proved that forcing Tensor Cores on extreme micro-blocks ($16 \times 16$) is an anti-pattern (SIMT was 17x faster). Tensor Cores require macro-blocks to amortize the ALU reconstruction overhead of the Ozaki INT8 scheme. 
Furthermore, the previous $O(4^N)$ GEMM was computationally explosive for $N \ge 14$. By decomposing the FWT via blocked Kronecker products, we retain the massive 660 TOPS INT8 throughput of the Tensor Cores while fundamentally shifting the algorithmic complexity to approach the theoretical $O(N \log N)$ of traditional FWT.

---

## 3. Environment
- **GPU**: NVIDIA RTX 4090 (Ada Lovelace / SM_89)
- **CUDA**: 12.x
- **Host**: Windows (WDDM Environment)
- **Peak Compute**: 660 TOPS (INT8) / 1.3 TFLOPs (FP64)
- **Peak Bandwidth**: 1008 GB/s

---

## 4. Workload
- **Operation**: IQP State Vector Encoding (FWHT)
- **Batch Size**: 128
- **Data Type**: FP64 (Reconstructed via INT8 Ozaki CRT Engine)
- **Scale Tested**: $N = 10, 12, 14, 15, 16$

---

## 5. Benchmark Methodology & Runtime Summary (Qubits N=14~16)

| Kernel | N | Time (ms) | Speedup vs PyTorch | Notes |
|----------|----------|----------|----------|----------|
| PyTorch Eager | 14 | 355.60 | 1.0x | Eager Baseline |
| Massive GEMM | 14 | 325.58 | 1.09x | $O(4^N)$ Tensor Core |
| **Blocked TC-FWT** | 14 | **14.41** | **24.6x** | $O(N \cdot 2^N)$ Tensor Core |
| PyTorch Eager | 15 | 1337.14 | 1.0x | 8.5GB VRAM Allocated |
| **Blocked TC-FWT** | 15 | **38.34** | **34.8x** | Bypassed WDDM Paging |
| PyTorch Eager | 16 | OOM | N/A | Exceeds 24GB VRAM |
| **Blocked TC-FWT** | 16 | **487.20** | **Infinity** | Matrix-Free Domination |

### Microkernel Benchmark (TDD-1 vs TDD-5, $N=4$, 16M elements)
| Architecture | Time (ms) | Speedup |
|----------|----------|----------|
| Tensor Core Ozaki | 27.89 | 1.0x |
| **SIMT Register FWT** | **1.63** | **17.1x** |

---

## 6. Performance Attribution & NCU Evidence

### A. Bottleneck Shift
- **Before (PyTorch/Naive FWT)**: Memory Bound (VRAM Allocation / Global Memory Traffic).
- **Before (Massive GEMM)**: Compute Bound (ALU Overhead from CRT Reconstruction).
- **After (Blocked TC-FWT)**: Tensor Core / Compute Bound (Highly optimized INT8 throughput with minimized CRT footprint).

### B. Warp Stall Analysis (`smsp__warp_issue_stalled*`)
- **Before (Branching `if`)**: High Branch Divergence Stalls during phase accumulation.
- **After (Branchless FMA)**: Branch Divergence eliminated. Stalls shifted to Instruction Fetch/Dependencies (expected in heavy ALU kernels).

### C. OS Level Memory Paging
- We discovered a catastrophic WDDM behavior: PyTorch's eager allocation causes massive VRAM fragmentation. Even after cache clearing, subsequent C++ processes at $N=15$ were forced to page to DDR5 System RAM over PCIe (Runtime spiked to 45,000 ms).
- **Fix**: The Matrix-Free nature of our engine completely bypasses eager VRAM allocation, ensuring pure HBM access when run independently.

---

## 7. Remaining Bottleneck & Next Step
**Current Bottleneck:** 
- Transpose Operations: The blocked TC-FWT requires intermediate batch transposes. Currently implemented naively in global memory.
- Single GPU Memory Limits: While Matrix-Free execution avoids $34GB$ matrices, the state vector itself will eventually exceed VRAM at $N \approx 28$.

**Next Step:**
- **Optimize Transpose**: Implement Shared Memory Conflict-Free Transpose kernels to fuse with the implicit Hadamard steps.
- **Stage 5 (Phase 2)**: Introduce NCCL / CUDA IPC to distribute the $N \ge 30$ state vector across multiple GPUs, implementing a Distributed Blocked TC-FWT.