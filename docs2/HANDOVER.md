# Handover：Phase 22 快路徑 + INT8 GEMM 第一刀

日期：2026-08-21  
卡：RTX 4060 Laptop，SM 8.9，CUDA 13.0  
分支：`main`（本 repo）

這份是 partial commit 的交接。完整帳在 [int8_gemm.md](int8_gemm.md)。  
iThome Day 8～10 怎麼切，在 `D:\D_backup\2025\tum\26w\Ithome\cASE\Blog\blueprint.md`（父專案，不在這個 git 裡）。

---

## 1. 現在可以相信的數字（2048³，`U(-1,1)`）

`examples/demo_main.cpp` 同一輪：native / cuBLAS Ozaki emu / Phase 22。

| | 時間 | vs native | max error vs native |
|---|---:|---:|---|
| cuBLAS native DGEMM | 82.2 ms | 1.0× | — |
| **Phase 22（這一刀之後）** | **10.9 ms** | **7.5×** | **3.40e-9** |
| cuBLAS 13 Ozaki emu | 8.5 ms | 9.6× | 2.84e-13 |
| 舊 Phase 24 ExtremeMix | ~130 ms | 0.6× | 3.55e-15（residual 重算，不要當標題） |

拆帳（第一次 execute）：`pack 1.66 ms  crt 9.38 ms`。  
`precision_test` 512：max error `1.52e-9`，過 1e-8。

**贏的是被限速的 native FP64。輸的是 cuBLAS emu（約 0.78×，誤差差四個數量級）。**

---

## 2. 這一刀改了什麼

### 預設路徑改 Phase 22

- `OzakiConfig.mode` 預設 `Phase22`，`enable_fp64` / `enable_residual` 預設 false
- demo、Python `AdaptiveGEMM(mode="phase22")`、precision_test 都走這條
- **不要再從 `Phase24ExtremeMix` 啟動**
- README 啟動範例已改；舊 1e-15 / 128 ms / lossless FP64 已刪

### Fast path（`AdaptiveOzakiEngine::execute`）

- INT8 CRT only：fused hi/lo modulo、不做 slice-max、不做 naive FP64 residual
- C 的 memset 在 default stream（修掉 non-blocking memset 跟下一輪 CRT 搶寫，那個 bug 會讓連續 execute 累加成剛好 4×）
- 比較 cuBLAS 時用 row-major `C=A*B` ≡ column-major `C=B*A`（`d_B, d_A` 對調）

### INT8 第一刀（CRT 裡的除法）

- 7 個質數 `% p` → `__umulhi` + magic（`d_prime_magic`）
- 滿 7 質數 `% M` → Barrett（`mu=109218`，`M=168897325606883`）
- 質數迴圈**沒有** unroll（裡面還有完整 K，展開會複製七份 MMA）
- wall time 13.6 ms → 10.9 ms，誤差數字不變（同一顆 worst element）

---

## 3. 不要踩的坑

1. **`CUBLAS_EMULATE_DOUBLE_PRECISION` 環境變數**會把 native DGEMM 也變成 emu。demo 開頭有清掉。
2. **legacy default stream 不會等 `cudaStreamNonBlocking`。** memset / CRT / 下一輪 execute 的依賴要自己接。
3. 拿 Phase 22 的 3e-9 去比 emu 的 3e-13 沒意義。公平賽是鎖 `cublasSetFixedPointEmulationMaxMantissaBitCount`。
4. Phase 24 residual 準是因為又算了一次 naive FP64 GEMM，而且比 native 慢。
5. WDDM 上 Ozaki MMA kernel 以前 NCU 會 `LaunchFailed`。本機 profile 先當可能打不開。

---

## 4. 下一個 session 做什麼

目標還是 `crt_fast_kernel` 的 INT8 MMA，不是再加 residual。

優先順序：

1. ~~smem XOR swizzle~~ 10.9 → 10.3 ms。
2. ~~質數 k=0 prefetch~~ 試過，CRT 變慢，已撤回。
3. ~~公平賽 mantissa~~ 已跑。34-bit cuBLAS 3.97 ms / 7.6e-10，Phase 22 約 11 ms / 3.4e-9，**同一精度預算輸 2.8×**。
4. **下一步：B.1/fused（把 INT32 scratch 砍掉）**。不要再微調這顆 CRT kernel 追那 2.8×。

成功標準：誤差維持 `~3e-9` / `1e-8` 以內；wall time 往 8.5 ms 靠近。不保證贏預設 emu。

重測：

```bat
cd case_2\adaptive_publish_tmp
nvcc --std=c++17 -O3 -arch=sm_89 -I include examples/demo_main.cpp src/AdaptiveOzaki.cu -o adaptive_gemm_demo.exe -lcublas -lcudart
nvcc --std=c++17 -O3 -arch=sm_89 -I include tests/precision_test.cpp src/AdaptiveOzaki.cu -o precision_test.exe -lcublas -lcudart
adaptive_gemm_demo.exe
precision_test.exe
```

改完把數字補進 `docs2/int8_gemm.md` 的重測紀錄。

---

## 5. 檔案地圖

| 檔 | 角色 |
|---|---|
| `src/AdaptiveOzaki.cu` | fast path + `crt_fast_kernel` + Barrett/magic |
| `include/AdaptiveOzaki.h` | 預設 Phase22、關 residual |
| `examples/demo_main.cpp` | 三方對打 |
| `src/pybind_module.cpp` | `AdaptiveGEMM(mode="phase22")` |
| `tests/precision_test.cpp` | 512、row-major cuBLAS |
| `README.md` | 對外數字與啟動方式 |
| `docs2/` | 這條線的開發記錄 |

父專案（`cASE`，另一個 git）：

- `Blog/blueprint.md`、`Blog/day_4.md`：Day 8 naive → Day 9 MMA → Day 10 吃滿
- 還沒寫 Day 6～10 正文
