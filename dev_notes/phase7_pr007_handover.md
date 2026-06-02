# Phase 7 Handover: IQP TC-FWT NCU + Performance Attribution (PR007)

## 1. 專案背景與目標 (Context & Goal)

Phase 4–5 已完成 Blocked TC-FWT 與 Implicit Ozaki Persistent Kernel。Phase 7 目標是讓 **PR 具備可審查的 NCU 證據**，對齊 `dev_notes/01pr_refference.md` 的 Performance Review Framework 與 `02pr_refference.md` 的 Engineering PR 要求。

**本階段交付：**
- 可重現的 benchmark + NCU 方法論（`reports/PR007_ncu_benchmark.md`）
- 文字化 Performance Attribution（本 handover）
- **可監測 kernel 路徑**（`OZAKI_NCU_PROFILE=1` → 非 persistent grid kernel，見 Section 6）

---

## 2. 環境 (Environment)

| 項目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 (SM 8.9, **WDDM**) |
| Driver | 581.57 |
| CUDA | 13.0 |
| NCU | 2024.3.0.0 (build 34567288) |
| OS | Windows（本機）/ WSL2 Ubuntu（已驗證） |
| WSL CUDA | 12.6（`/usr/local/cuda`），NCU 2024.3.2 |

> WDDM / WSL2 下 **Ozaki MMA kernel**（含 grid profiling 路徑）在 NCU 中皆 `LaunchFailed`；side kernel 可 profile。見 **Section 11**。

---

## 3. Workload

| 參數 | N=14 | N=16 |
|------|------|------|
| Qubits | 14 | 16 |
| State len | 16384 | 65536 |
| Batch (bench) | 128 | 128 |
| Batch (NCU) | 1 | 1 |
| Datatype | FP64 state, INT8 Ozaki tiles | 同左 |
| Correctness ref | `launch_iqp_encode_batch` (global FWT) | 同左 |

---

## 4. Q1 — 快多少？ (Runtime Summary)

**測量：** `bench_kernels_perf.exe`（無 register cap），計時 **TC path only**（`launch_iqp_encode_tc`）。

| Kernel / Path | N | Batch | Time (µs) | vs Baseline FWT* | Max Abs Error |
|---------------|---|-------|-----------|------------------|---------------|
| Baseline FWT (`launch_iqp_encode_batch`) | 14 | 128 | ~included in verify only | 1.00× | 0 |
| **TC path** (`launch_iqp_encode_tc`) | 14 | 128 | **93** | see note | **0** |
| **TC path** | 16 | 128 | **109** | see note | **0** |

\*Bench 驅動以 baseline 做 correctness；端到端 baseline 計時需另跑 `bench_kernels` 擴充。NCU 顯示 baseline butterfly stage ~3.18 µs × 14 stages ≈ 45 µs order（單 sample），與 TC 93 µs 屬不同 pipeline（TC 含 phase split + 4× Ozaki + transpose）。

**結論（Q1）：** TC path 在 N=14/16、batch=128 下 **數值零誤差通過**；絕對延遲 93/109 µs（2026-06-02 重測，較初版 75/92 µs 有 run-to-run 波動）。

---

## 5. Q2 — 為什麼快 / 路徑差異？ (Performance Attribution)

| 因素 | Baseline FWT | TC path (side kernels + Ozaki) |
|------|--------------|--------------------------------|
| 演算法 | Global-memory butterfly | Kronecker + implicit H (no H matrix storage) |
| Tensor Core | 無 | Ozaki INT8 MMA (`mma.m16n8k32`) |
| Memory | DRAM ~33% (butterfly) | Phase split: DRAM ~0.4%（compute-bound） |
| Occupancy | Warps active ~22% | Phase split ~38%；modulo precompute ~56% |

**Partial NCU evidence（N=14, batch=1）：**

| Kernel | DRAM % | L2 % | SM % | Warps active % | Reg/thread | Duration |
|--------|--------|------|------|----------------|------------|----------|
| `fwt_butterfly_batch_kernel` (avg/stage) | 33.5 | 10.5 | 10.3 | 21.7 | 28 | 3.18 µs |
| `iqp_phase_split_kernel` | 0.4 | 2.3 | **67.5** | 37.5 | 60 | 12.74 µs |
| `precompute_modulo_kernel_p26_implicit` | 18.8 | 5.6 | 17.1 | 56.2 | 28 | 2.88 µs |
| `implicit_hadamard_ozaki_grid_kernel_implicit` (`OZAKI_NCU_PROFILE=1`) | — | — | — | — | — | **LaunchFailed under NCU on WDDM**; **PASSED** outside NCU |
| `implicit_hadamard_ozaki_persistent_kernel_implicit` (default prod) | — | — | — | — | — | Not used when profiling env set |

**Artifacts：** `qdp/qdp-kernels/ncu_summary.txt`，`qdp/qdp-kernels/reports/ncu_*_clean.csv`

---

## 6. Q3 — 目前 Bottleneck (Bottleneck Analysis)

| 元件 | 判斷 | 依據 |
|------|------|------|
| Baseline FWT | **Memory-leaning** | DRAM ~33%，SM ~10% |
| Phase split | **Compute-bound** | SM ~67%，DRAM ~0.4% |
| Modulo precompute | Mixed (mem + compute) | DRAM ~19%，SM ~17% |
| Ozaki MMA (persistent) | 無法在 WDDM+NCU 量測 | LaunchFailed |
| Ozaki MMA (grid, profiling) | 預期 **Tensor / register bound** | 待 `sm__pipe_tensor_active` |

**Bottleneck shift（可寫進 PR）：**
- Butterfly：**Memory / low SM util** → Phase split：**Compute-bound**
- 核心 TC GEMM：需 grid profiling kernel 補 **Tensor Core Active %**

---

## 7. Q4 — 下一步 (Next Steps)

1. **已完成：** `OZAKI_NCU_PROFILE=1` → `implicit_hadamard_ozaki_grid_kernel_implicit`（static smem，一 tile 一 block）；正確性與 production persistent 路徑一致（batch=1 驗證 MAE=0）。
2. **WDDM NCU：** grid kernel 在 NCU 下仍 `LaunchFailed`（inline MMA 限制）；評測請用 side kernels NCU + 一般 bench，或 **Linux/TCC** 跑完整 Ozaki NCU。
3. 補 **Warp stall** / **bank conflict** metrics（01 framework Section 9–11）。
4. **裸機 Linux**（非 WSL/Docker）上重試 Ozaki grid NCU + `sm__pipe_tensor_active`（見 Section 11）。
5. PR 合併前：補 baseline vs TC **端到端** speedup 表。

---

## 8. 關鍵文件導覽 (Key Files)

| 路徑 | 角色 |
|------|------|
| `reports/PR007_ncu_benchmark.md` | NCU 命令、結果、失敗分析 |
| `reports/ncu_summary.txt` | 一頁 NCU 摘要 |
| `dev_notes/01pr_refference.md` | Reviewer 要求的 NCU 模板 |
| `dev_notes/02pr_refference.md` | PR 說服力檢查清單 |
| `qdp/qdp-kernels/src/ImplicitHadamardOzaki.cu` | Persistent + **grid profiling** kernel |
| `qdp/qdp-kernels/tests/bench_kernels.cu` | Bench + `IQP_NUM_SAMPLES` |
| `qdp/qdp-kernels/scripts/wsl_build_and_ncu.sh` | WSL 一鍵編譯 + NCU 嘗試 |

---

## 9. 給接手者：NCU 怎麼跑 (Repro)

```powershell
# Build (profiling binary)
nvcc -O3 -std=c++17 -arch=sm_89 --maxrregcount=64 -I qdp/qdp-kernels/src `
  -o qdp/qdp-kernels/build/bench_kernels.exe `
  qdp/qdp-kernels/tests/bench_kernels.cu ... (see PR007)

# Side kernels
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,launch__shared_mem_per_block,launch__registers_per_thread `
  --kernel-name iqp_phase_split_kernel --csv qdp/qdp-kernels/build/bench_kernels.exe 14 1

# Ozaki grid (profiling path)
$env:OZAKI_NCU_PROFILE = "1"
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,smsp__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,launch__shared_mem_per_block,launch__registers_per_thread `
  --kernel-name implicit_hadamard_ozaki_grid_kernel_implicit --csv qdp/qdp-kernels/build/bench_kernels.exe 14 1
```

**WSL2（Ubuntu）：**

```powershell
wsl -d Ubuntu -- bash /mnt/d/D_backup/2025/tum/26S/apache_mout/qdp/qdp-kernels/scripts/wsl_build_and_ncu.sh
```

產物：`build_wsl/bench_kernels`、`reports/ncu_ozaki_grid_wsl.csv`（Ozaki 仍可能 LaunchFailed）。

---

## 10. PR Review 對照 (02pr_refference checklist)

| 要求 | 狀態 |
|------|------|
| Baseline 定義 | 有（batch FWT）；端到端 speedup 表待補 |
| NCU / Occupancy / DRAM | Side kernels 完成（Windows + WSL）；Ozaki MMA NCU 失敗 |
| Tensor Utilization | 待裸機 Linux；WSL/Docker 不保證 |
| Accuracy (max abs error) | **0** @ N=14,16 |
| Roofline / Warp stall | 未做（下一 PR 可補） |
| 避免 peak TOPS 空話 | PR007 未使用 |

---

## 11. WSL2 / Docker 實測結論（2026-06-02）

### 11.1 WSL2 Ubuntu 測試

| 項目 | 結果 |
|------|------|
| GPU / Driver | `nvidia-smi` 正常（Driver 581.57，CUDA 13.0 API） |
| 編譯 + bench（`OZAKI_NCU_PROFILE=1`） | **PASSED**，MAE=0 |
| NCU `iqp_phase_split_kernel` | **成功**（例：SM ~67%，DRAM ~0.4%） |
| NCU `implicit_hadamard_ozaki_grid_kernel_implicit` | **`LaunchFailed`**（與 Windows 相同） |

**結論：** WSL2 **不是**「換個 Linux 就能量 Ozaki MMA」。它對 **bench / 正確性 / side-kernel NCU** 很有用；對 **inline MMA + 大 smem** 的完整 NCU，與 Windows 一樣仍卡住（WSL2 GPU 仍經 Windows 顯示驅動，`Disp.A On`）。

### 11.2 Docker 會不會不一樣？

**在 Windows 上開 Docker Desktop → 多半一樣。**

| 場景 | 預期 |
|------|------|
| Docker Desktop（WSL2 後端 + GPU） | 與 WSL2 類似：跑 bench 可以，Ozaki MMA 的 NCU **很可能仍 LaunchFailed** |
| Docker 內只跑 `bench_kernels`、不跑 NCU | 可以，適合 CI/他人重現數值 |
| Docker 內跑 `ncu` | 容器需 `--gpus all` + 驅動/toolkit 掛載；**不保證**比 WSL 更好 |
| 雲端 / 實驗室 **裸機 Linux**（無桌面 GPU） | 唯一仍值得認真嘗試的完整 NCU 路線 |

Docker **不能**繞過「這顆 kernel 對 NCU replay 不友善」這件事；它只是多一層隔離，底層還是你的 4060 + Windows 驅動棧。

### 11.3 覺得太難時，PR 怎麼交差（務實）

Reviewer 要的是 **Performance Attribution**，不是「每顆 kernel 都有 NCU」：

1. **必交：** bench 表 + MAE=0 + side-kernel NCU（phase split / modulo / FWT）+ bottleneck 文字（handover Q2–Q3）。
2. **誠實註明：** Ozaki MMA 在 Windows/WSL/Docker+4060 下 NCU `LaunchFailed`；已提供 `OZAKI_NCU_PROFILE` grid 路徑供執行與對照。
3. **可選加分：** 實驗室裸機 Linux 補一張 `sm__pipe_tensor_active`（不是 PR blocker，若導師堅持再排期）。

---

**Handover status:** NCU partial（side kernels OK on Win+WSL）| Grid profiling kernel shipped | Ozaki MMA NCU blocked on consumer GPU + host driver stack  
**Updated:** 2026-06-02（含 WSL2 實測）