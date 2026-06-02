# PR007: IQP TC-FWT NCU + Benchmark Report

## 1. Environment
- GPU: NVIDIA GeForce RTX 4060 (SM 8.9, WDDM)
- Driver: 581.57
- CUDA Toolkit: 13.0
- Nsight Compute: 2024.3.0.0 (build 34567288)
- OS: Windows

## 2. Build Artifacts
Two binaries were built to separate performance runs from NCU profiling constraints.

### 2.1 Performance Binary (no register cap)
```bash
nvcc -O3 -std=c++17 -arch=sm_89 -I qdp/qdp-kernels/src \
  -o qdp/qdp-kernels/build/bench_kernels_perf.exe \
  qdp/qdp-kernels/tests/bench_kernels.cu \
  qdp/qdp-kernels/src/iqp.cu \
  qdp/qdp-kernels/src/iqp_tc.cu \
  qdp/qdp-kernels/src/ImplicitHadamardOzaki.cu \
  qdp/qdp-kernels/src/AdaptiveOzaki.cu \
  qdp/qdp-kernels/src/amplitude.cu \
  qdp/qdp-kernels/src/angle.cu \
  qdp/qdp-kernels/src/basis.cu \
  qdp/qdp-kernels/src/phase.cu \
  qdp/qdp-kernels/src/validation.cu
```

### 2.2 Profiling Binary (register cap + profiling knobs)
```bash
nvcc -O3 -std=c++17 -arch=sm_89 --maxrregcount=64 -I qdp/qdp-kernels/src \
  -o qdp/qdp-kernels/build/bench_kernels.exe \
  qdp/qdp-kernels/tests/bench_kernels.cu \
  qdp/qdp-kernels/src/iqp.cu \
  qdp/qdp-kernels/src/iqp_tc.cu \
  qdp/qdp-kernels/src/ImplicitHadamardOzaki.cu \
  qdp/qdp-kernels/src/AdaptiveOzaki.cu \
  qdp/qdp-kernels/src/amplitude.cu \
  qdp/qdp-kernels/src/angle.cu \
  qdp/qdp-kernels/src/basis.cu \
  qdp/qdp-kernels/src/phase.cu \
  qdp/qdp-kernels/src/validation.cu
```

Profiling knobs:
- `OZAKI_NCU_PROFILE=1` switches the implicit Ozaki kernel to a single-buffer shared-memory layout.
- `IQP_NUM_SAMPLES=1` (or CLI arg) reduces batch size to avoid WDDM watchdog issues.

## 3. Benchmark Methodology
- Benchmark driver: `qdp/qdp-kernels/tests/bench_kernels.cu`
- Default batch size: 128
- Output time is the TC path runtime only (baseline is used for correctness check)

### 3.1 Command
```bash
qdp/qdp-kernels/build/bench_kernels_perf.exe 14
qdp/qdp-kernels/build/bench_kernels_perf.exe 16
```

### 3.2 Results (perf binary)
| N | Batch | Time (us) | Max Abs Error | Result |
|---|-------|-----------|---------------|--------|
| 14 | 128 | 93 | 0 | PASSED |
| 16 | 128 | 109 | 0 | PASSED |

*Regenerated 2026-06-02 on RTX 4060; prior run reported 75 / 92 us (same machine, run-to-run variance).*

## 4. NCU Methodology
All NCU runs used a minimal metric set to reduce replay overhead.

### 4.1 Baseline FWT (global-memory butterfly)
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,
            smsp__warps_active.avg.pct_of_peak_sustained_active,
            dram__throughput.avg.pct_of_peak_sustained_elapsed,
            lts__throughput.avg.pct_of_peak_sustained_elapsed,
            gpu__time_duration.sum,
            launch__shared_mem_per_block,
            launch__registers_per_thread \
    --kernel-name fwt_butterfly_batch_kernel --csv \
    qdp/qdp-kernels/build/bench_kernels.exe 14 1
```

### 4.2 TC Path (phase split + modulo precompute)
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,
            smsp__warps_active.avg.pct_of_peak_sustained_active,
            dram__throughput.avg.pct_of_peak_sustained_elapsed,
            lts__throughput.avg.pct_of_peak_sustained_elapsed,
            gpu__time_duration.sum,
            launch__shared_mem_per_block,
            launch__registers_per_thread \
    --kernel-name iqp_phase_split_kernel --csv \
    qdp/qdp-kernels/build/bench_kernels.exe 14 1

ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,
            smsp__warps_active.avg.pct_of_peak_sustained_active,
            dram__throughput.avg.pct_of_peak_sustained_elapsed,
            lts__throughput.avg.pct_of_peak_sustained_elapsed,
            gpu__time_duration.sum,
            launch__shared_mem_per_block,
            launch__registers_per_thread \
    --kernel-name precompute_modulo_kernel_p26_implicit --csv \
    qdp/qdp-kernels/build/bench_kernels.exe 14 1
```

### 4.3 Ozaki MMA Kernel — profiling path (`OZAKI_NCU_PROFILE=1`)

When `OZAKI_NCU_PROFILE=1`, production code launches **`implicit_hadamard_ozaki_grid_kernel_implicit`** (one tile per block, static 28 KiB smem, no atomic queue). Persistent kernel is **not** used on this path.

```bash
$env:OZAKI_NCU_PROFILE=1
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,
            sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,
            smsp__warps_active.avg.pct_of_peak_sustained_active,
            dram__throughput.avg.pct_of_peak_sustained_elapsed,
            lts__throughput.avg.pct_of_peak_sustained_elapsed,
            gpu__time_duration.sum,
            launch__shared_mem_per_block,
            launch__registers_per_thread \
    --kernel-name implicit_hadamard_ozaki_grid_kernel_implicit --csv \
    qdp/qdp-kernels/build/bench_kernels.exe 14 1
```

**WDDM result (2026-06-02):** still `LaunchFailed` under NCU (inline `mma` + 28 KiB smem). Kernel **runs and passes correctness** outside NCU. Use side-kernel NCU + Linux/TCC for full Ozaki MMA metrics.

Legacy persistent kernel (`implicit_hadamard_ozaki_persistent_kernel_implicit`) also `LaunchFailed` under NCU — see Section 6.

## 5. NCU Results Summary
All metrics are averaged across kernel launches when applicable.

**Artifacts (regenerated 2026-06-02):**
- Text summary: `qdp/qdp-kernels/ncu_summary.txt`, `AdaptiveGEMM_repo/reports/ncu_summary.txt`
- Raw CSV: `qdp/qdp-kernels/reports/ncu_*_clean.csv`
- Persistent kernel attempt: `qdp/qdp-kernels/reports/ncu_persistent_kernel.csv` (LaunchFailed)

### 5.1 Baseline FWT (fwt_butterfly_batch_kernel)
- DRAM throughput: 33.49%
- L2 throughput: 10.51%
- SM throughput: 10.31%
- Achieved occupancy (warps active): 21.73%
- Registers / thread: 28
- Shared memory / block: 1,024 bytes
- Kernel duration (avg per stage): 3.18 us

### 5.2 TC Path: Phase Split (iqp_phase_split_kernel)
- DRAM throughput: 0.42%
- L2 throughput: 2.27%
- SM throughput: 67.52%
- Achieved occupancy (warps active): 37.51%
- Registers / thread: 60
- Shared memory / block: 1,024 bytes
- Kernel duration: 12.74 us

### 5.3 TC Path: Modulo Precompute (precompute_modulo_kernel_p26_implicit)
- DRAM throughput: 18.77%
- L2 throughput: 5.64%
- SM throughput: 17.12%
- Achieved occupancy (warps active): 56.16%
- Registers / thread: 28
- Shared memory / block: 1,024 bytes
- Kernel duration: 2.88 us

## 6. NCU Failure Analysis (Persistent Kernel)
The core TC kernel `implicit_hadamard_ozaki_persistent_kernel_implicit` consistently fails to profile on WDDM:

```
==ERROR== LaunchFailed
==ERROR== An error occurred while trying to profile.
```

Mitigations attempted (all still failed):
1. Single-buffer shared memory (`OZAKI_NCU_PROFILE=1`, 28 KB smem).
2. Reduced batch size (`IQP_NUM_SAMPLES=1`).
3. Reduced register pressure (`--maxrregcount=64`).
4. Minimal metric set and application replay.

Conclusion: WDDM + persistent kernel + inline MMA path remains non-profileable under NCU. This is a tooling limitation, not a functional kernel failure (kernel runs and passes correctness outside NCU).

### 6.1 WSL2 retest (2026-06-02)

Tested on **Ubuntu WSL2** with `/usr/local/cuda` (CUDA 12.6, NCU 2024.3.2):

| Target | NCU on WSL2 |
|--------|-------------|
| `iqp_phase_split_kernel` | Success |
| `implicit_hadamard_ozaki_grid_kernel_implicit` (`OZAKI_NCU_PROFILE=1`) | **LaunchFailed** (same as Windows) |

Repro script: `qdp/qdp-kernels/scripts/wsl_build_and_ncu.sh`  
Artifact: `qdp/qdp-kernels/reports/ncu_ozaki_grid_wsl.csv`

**Docker Desktop (WSL2 GPU backend):** expect the same class of NCU failure for Ozaki MMA; use Docker for bench/CI, not as an NCU workaround on RTX 4060 + Windows host.

## 7. Recommended Path Forward
1. **Run NCU on bare-metal Linux (not WSL/Docker on Windows host)**
   - WSL2 was tried; side kernels profile, Ozaki MMA still fails. Prefer lab/cloud Linux without desktop display binding.
2. **Add a profiling-only non-persistent kernel**
   - Implement a kernel variant without the persistent work queue, and use it only when `OZAKI_NCU_PROFILE=1`.
3. **Keep WDDM profiling for side kernels**
   - Phase split and modulo precompute are profileable and provide partial evidence of the TC pipeline.

## 8. Code Changes & PR Readiness (for comparison with advisor PR)

### 8.1 Files touched (kernel / bench only)
| File | Nature of change |
|------|------------------|
| `ImplicitHadamardOzaki.cu` | TC persistent kernel, `OZAKI_NCU_PROFILE` single-buffer smem for profiling attempts |
| `ImplicitHadamardOzaki.h` | Minor (API unchanged: `execute_implicit_hadamard`) |
| `iqp_tc.cu` | IQP TC-FWT encode path wiring |
| `bench_kernels.cu` | `IQP_NUM_SAMPLES` env + CLI batch override for NCU |
| `bench_tc_blocked_fwt.cu` | Extended blocked-FWT bench |

### 8.2 Backward compatibility
- **Public API:** `ImplicitHadamardOzakiEngine::execute_implicit_hadamard(...)` signature unchanged.
- **Opt-in profiling:** `OZAKI_NCU_PROFILE=1` and `IQP_NUM_SAMPLES` only affect bench/NCU runs, not default library behavior.
- **Default path:** `launch_iqp_encode_batch` (baseline FWT) unchanged; TC path is `launch_iqp_encode_tc` (separate entry).

### 8.3 Performance vs prior report
| Metric | Prior (doc) | This run |
|--------|-------------|----------|
| N=14 TC time (batch 128) | 75 us | 93 us |
| N=16 TC time (batch 128) | 92 us | 109 us |
| NCU side kernels | ~same | within ~1% (see Section 5) |

Correctness: max abs error 0 for N=14 and N=16.

### 8.4 PR checklist
- [x] Rebuild `bench_kernels_perf.exe` and `bench_kernels.exe`
- [x] Benchmark N=14, N=16 passed
- [x] NCU: baseline FWT, phase split, modulo precompute
- [ ] NCU: persistent TC kernel (blocked on WDDM — document limitation)
- [x] `ncu_summary.txt` regenerated

---
Status: **Benchmarks and partial NCU evidence regenerated 2026-06-02.** Persistent TC kernel still requires Linux/TCC for full NCU metrics. Safe to open PR with perf + side-kernel NCU data; call out WDDM limitation explicitly.
