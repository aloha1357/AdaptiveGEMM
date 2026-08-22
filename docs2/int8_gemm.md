# INT8 GEMM：從「會算」到「吃得滿」

最後更新：2026-08-21  
卡：RTX 4060 Laptop，SM 8.9，CUDA 13.0  
矩陣：2048³，`U(-1,1)` float64

---

## 一句話

Phase 22 的數學是對的，end-to-end 也比 native `cublasDgemm` 快。
現在輸 cuBLAS 13 Ozaki emu，主要不是 Ozaki 公式，是 **INT8 乘法這段沒吃滿 Tensor Core**。

Day 8～10 就寫這件事。一天只改一層，先對答案再談快。

---

## 現在站在哪

同一輪、同一組資料（demo `examples/demo_main.cpp`，Phase 22）：

| | 時間 | 等效 GFLOPS | vs native | max error vs native |
|---|---:|---:|---:|---|
| cuBLAS native DGEMM | 82.2 ms | 209 | 1.0× | — |
| AdaptiveGEMM Phase 22（XOR 後） | 10.3–11.2 ms | ~1540–1660 | ~7.5× | 3.40e-9 |
| cuBLAS 13 Ozaki emu（dynamic） | 8.5–8.8 ms | ~1940–2010 | ~9.5× | 2.84e-13 |
| cuBLAS emu **34-bit fixed** | **3.97 ms** | ~4330 | ~21× | **7.62e-10** |

1-pass（只做 hi×hi）大約 4.4 ms，誤差已經接近完整 4-pass 的 `1e-9`。
4-pass 13.6 ms ≈ pack ~1.4 ms + 4 × ~3.1 ms CRT。

粗帳（把 4 個 slice 乘積 × 7 個質數看成 28 次 2048³ INT8 GEMM）：

| | 每次 INT8 GEMM | 等效 TOPS | 對 4060 dense INT8 峰值 |
|---|---:|---:|---|
| 我們 `crt_fast_kernel` | ~0.43 ms | ~40 | ~30% |
| cuBLAS emu（估 ~49 次 8-bit GEMM 打進 8.5 ms） | ~0.17 ms | ~100 | ~80% |

舊 NCU（`decoupled_crt_pass_kernel`，不是 end-to-end）：

- Tensor (INT8) pipe ~36%
- Occupancy ~48%（`__launch_bounds__(256, 3)`）
- Shared memory bank conflict 仍高
- 那份報告把 130 ms 的 Phase 24 寫成「kernel 已經不錯」——CRT kernel 確實不是 130 ms 的主因，naive residual 才是。**現在 residual 拿掉了，CRT 裡的 INT8 MMA 變成主因。**

---

## 兩層不要混

1. **MMA / tiling / smem 沒吃滿**（這次要優化的）  
   同樣是 `mma.sync.m16n8k32.s8`，我們大約 30% 峰值，NVIDIA 大約 80%。

2. **架構比較肥**  
   我們是 17-bit × 2 slice，乘完 ×K 超過 int32，所以 **每次 slice 乘積裡面還要跑 7 質數 CRT**。  
   cuBLAS 是 8-bit multi-slice（Scheme I），一次 INT8 GEMM 的乘積塞得進 int32，**不需要**我們這種 CRT。  
   slice 數 `ceil((mantissaBits+1)/8)`，成本跟 slice **平方**成長。

所以：kernel 做好，34-bit（~1e-9）有機會靠近 emu。  
要同時準到 1e-13，在這張 4060 上不當成目標。

公平賽已跑：見下方「fixed mantissa」表。34-bit 對 34-bit，cuBLAS 仍快約 2.8×。

---

## 對上 Day 8 / 9 / 10

原本藍圖 Day 8 求餘數、Day 9 重建、Day 10 fused。那條路還在，只是 **8～10 改成圍著 INT8 乘法轉**，因為量完才知道瓶頸在這裡。

| 天 | 只問一句 | 工程上做什麼 | 結束時要有的東西 |
|---|---|---|---|
| **Day 8** | 7 張 INT8 矩陣，最笨的乘法怎麼寫才是對的？ | naive INT8 GEMM；`precompute_modulo` 當前置；跟 CPU `A8 @ B8` 對餘數 | 答案對、很慢。讀者知道「這就是要丟進 Tensor Core 的東西」 |
| **Day 9** | 怎樣才走 `mma.sync`？ | 128×64 tile、`cp.async` 雙緩衝、`ldmatrix`、K=32 的 INT8 MMA；7 個質數還是序列 | 現在這顆 `crt_fast_kernel` 的骨架。誤差 ~1e-9，13 ms 量級 |
| **Day 10** | 為什麼 tensor pipe 只有 36%？能不能靠近 cuBLAS？ | bank conflict、occupancy、質數迴圈、`% p` / `% M` 換成乘法、對打 emu | 一輪優化數字。不假裝贏 NVIDIA |

Day 6、Day 7 不動（三步圖、CRT 表）。求餘數改成 Day 8 開頭帶過：沒有 INT8 矩陣就沒有 GEMM。

---

## Kernel 要動的地方

目標檔：`src/AdaptiveOzaki.cu` 裡的 `crt_pass_kernel_body` / `crt_fast_kernel`。

現況（Day 9 骨架）：

- tile 128×64，8 warp，每 warp 32×32
- K 方向 32，`cp.async` 雙緩衝
- 每個質數跑完整個 K，再 `cf % p`、乘 `f_p`、`% M`
- 4 個 slice 乘積 = 4 次 launch，C 用 `+=`（tile 不重疊，不用 atomic）

第一批（先做、風險低）：

1. **7 個質數的 `% p` 改乘法**  
   accumulator 上限 `126*126*2048 = 32514048`，用 `__umulhi` + 預計算 magic。
2. **CRT 的 `% M` 改 Barrett**（只在滿 7 質數、`M = 168897325606883` 時）  
   `mu = floor(2^64 / M) = 109218`，`__umul64hi`。`nl < 7` 的舊路徑維持 `% M`。
3. **質數迴圈展開先不做**  
   裡面還有完整 K 迴圈，展開 7 次會把 MMA 本體複製七份，I-cache / 編譯都痛。等 bank conflict 做完再看要不要只 pipeline「下一個質數的第一塊 tile」。

下一批（Day 10，還沒做）：

4. smem XOR swizzle，打 bank conflict（ldmatrix 現在是 row-major + pad 48）
5. 下一個質數的 k=0 tile 提前 `cp.async`
6. 公平賽：cuBLAS 鎖 mantissa bits
7. 若 34-bit 仍輸 3×，再評估要不要改 8-bit Scheme I（那是另一條架構，不是這顆 kernel 的微調）

不要做：

- 把 Phase 24 residual 加回來衝 1e-15
- 加第三片 17-bit 去追 1e-13（pass 從 4 變 9，會更慢）
- 在還沒對答案前改 ldmatrix swizzle

---

## 重測紀錄

每次改 kernel 都跑 `adaptive_gemm_demo.exe`（native / emu / Phase 22）和 `precision_test.exe`（512，tolerance 1e-8）。

### 2026-08-21 baseline（優化前）

| | ms | err |
|---|---:|---|
| native | 82.2 | — |
| Phase 22 | 13.6 | 3.40e-9 |
| cuBLAS emu | 8.5 | 2.84e-13 |
| precision_test 512 | — | 1.52e-9 |

### 2026-08-21 after magic rem + Barrett

`% p` → `__umulhi` magic；滿 7 質數的 `% M` → Barrett `__umul64hi`。質數迴圈沒展開。誤差跟 baseline 同一顆元素、同一個 3.39771e-9，所以是同一條數學。

| | ms | err |
|---|---:|---|
| native | 82.2 | — |
| Phase 22 | **10.94** | 3.40e-9 |
| cuBLAS emu | 8.52 | 2.84e-13 |
| precision_test 512 | — | 1.52e-9 |

第一次 execute 拆帳：`pack 1.66 ms  crt 9.38 ms  total 11.05 ms`。  
CRT 4 pass ≈ 2.35 ms/pass（原本 ~3.1 ms）。Wall time 13.6 → 10.9 ms（約 20%）。對 emu 從 0.63× 收到 0.78×。

還差 ~2.4 ms 才跟預設 emu 同一量級。下一步是 smem bank conflict / ldmatrix layout，不是再砍除法。

### 2026-08-21 after smem XOR swizzle

`sa/sb` pad 48→64，store / `ldmatrix` 用 `col ^ ((row & 3) << 4)`。誤差不變。

| | ms | err |
|---|---:|---|
| native | 82.2 | — |
| Phase 22 | **10.33** | 3.40e-9 |
| cuBLAS emu | 8.52 | 2.84e-13 |

CRT 9.38 → 8.86 ms。Wall 10.94 → 10.33 ms（再 6%）。對 emu 0.78× → 0.82×。

Bank conflict 沒有一次打掉。

### 2026-08-21 prefetch next prime：撤回

最後一個 K tile 預取下一個質數的 k=0。答案仍對，CRT 8.86 → 10.36 ms，wall 10.3 → 11.8 ms。質數之間 CRT 跟 load 重疊蓋不過額外的指標／分支。已 revert。

### 2026-08-21 公平賽：cuBLAS fixed mantissa

`CUDA_EMULATION_MANTISSA_CONTROL_FIXED` + `MaxMantissaBitCount`。slice ≈ `ceil((bits+1)/8)`。

| mantissa | 估 slice / GEMM 次數 | ms | max err vs native | vs Phase 22 |
|---|---:|---:|---:|---|
| 16 | 3 / 9 | 2.24 | 4.26e-5 | emu 快 5.0× |
| 24 | 4 / 16 | 2.99 | 1.74e-7 | emu 快 3.7× |
| **34** | **5 / 25** | **3.97** | **7.62e-10** | **emu 快 2.8×** |
| 53 | 7 / 49 | 6.63 | 2.84e-13 | emu 快 1.7× |
| dynamic（預設） | ADP | 8.85 | 2.84e-13 | emu 快 1.3× |

對等 34-bit：cuBLAS **3.97 ms / 7.6e-10**，我們 **11.2 ms / 3.4e-9**。  
不是「差一點」，是同一精度預算下大約 **2.8×**。下一步若還要追，該評估 8-bit Scheme I（換架構），不是再微調這顆 CRT-in-MMA kernel。

---

## 什麼叫這輪成功

- 誤差不能壞：2048 max error 仍在 `1e-8` 以內，理想維持 `~3e-9`。
- Phase 22 wall time 往 10 ms 靠近。到 8.5 ms 才是「跟預設 emu 同一量級」，那是 Day 10 的天花板，不是今天。
- Demo 預設維持 Phase 22。README 的啟動方式不准寫回 Phase 24。

---

## PR1：孤立 s8×s8→s32 GEMM（Stage A，正確性第一）

**新增**：`include/mma_sm89.cuh`、`include/int8_gemm.h`、`src/int8_gemm.cu`  
**測試**：`tests/int8_gemm_test.cpp` → `adaptive_gemm_int8_test`  
**Demo 新增**：`[INT8 GEMM]` 列、GPU 屬性（clockRate / memoryBusWidth / sharedMemPerMultiprocessor）

### Kernel 規格（`<128,128,64,3>`）

| 項目 | 值 |
|------|---|
| CTA tile | BM=128, BN=128, BK=64 |
| Stages | 3（3-stage cp.async pipeline） |
| Threads | 256（8 warps） |
| Warp map | wm=wid/2, wn=wid%2; warp tile 32×64 |
| smem A | `[3][128][128]` = 48 KB（stride 128 for XOR） |
| smem B | `[3][128][64]` = 24 KB |
| smem total | **96 KB** / 101376 B 上限 → 1 CTA/SM (for `<128,256,64,3>`) |
| XOR swizzle | `col_b ^ ((row & 7) << 4)`；A `col_b∈[0,64)`，B `col_b∈[0,128)` |
| ldmatrix A | `m8n8.x4`（無 .trans） |
| ldmatrix B | `m8n8.x2` |
| RF frag | `a_frag[2][BM/64][4]`、`b_frag[2][BN/16][2]`（雙緩衝） |
| Inner loop | BM/64 × BN/16 mma.m16n8k32 per k1 step |
| 對齊契約 | `m%128==0, n%128==0, k%64==0`；否則 `cudaErrorInvalidValue` |

### TOPS vs cuBLAS INT8（2048³，RTX 4060 Laptop）

> **注意**：以下數字在 PR3 合入後首次執行 `adaptive_gemm_demo` 填入。

| Kernel | ms | TOPS | vs 121 TOPS | vs cuBLAS INT8 |
|--------|---:|-----:|------------:|---------------:|
| **int8_gemm_s8s8s32 `<128,256,64,3>`** | 0.76339 | 22.50 | 18.59% | 90.37% |
| `cublasGemmEx(CUDA_R_8I, COMPUTE_32I)` | 0.68994 | 24.90 | 20.57% | 100.00% |

- Gate A1：≥70% × 121 TOPS（≥85 TOPS）或 ≥90% × 實測 cuBLAS INT8 → **PR3**
- Gate A2：≥80% × 121 TOPS（≥97 TOPS）或 ≥95% × 實測 cuBLAS INT8 → **PR3b**
- PR1 速度可以很差；正確性（bit-exact vs cuBLAS INT8）是此 PR 的唯一 gate。

### Phase 22 回歸

`precision_test` n=512 仍綠（tolerance 1e-8）；Phase 22 不受 PR1 / PR2 影響。

---

## PR2：Scheme I B v0（cuBLAS INT8 + INT32 scratch）

**新增**：`include/scheme_i.h`、`src/scheme_i.cu`、`tests/scheme_i_precision_test.cpp`  
**Demo**：`[SchemeI] bits=34 p=5 pairs=15 backend=cublas`

正確性閘 only，**無 6.0 ms 閘**。B v0 估 1760 MiB HBM（INT32 落地）。Default 仍 Phase 22。

n=512 vs CPU gold（mantissa=34, p=5）：

| 資料 | 15 pair（T=4，預設） | 25 pair（oracle） |
|------|---------------------:|------------------:|
| U(-1,1) | 2.07e-6 | **1.41e-9** |
| mixed（一列 \|A\|~256） | 5.07e-4 | 2.27e-7 |

- 25 pair + U(-1,1) ≤ 1e-8：編碼 / bilinear bias / cuBLAS INT8 是通的。
- 15 pair ~2e-6 是丟 `a+b≥5` 的代價，不是 bug。T=4 是速度預設，不是 1e-8。
- mixed 25 pair 絕對誤差未到 1e-8（2.3e-7），文件照實寫。
- k=1 golden 走 naive INT8（cuBLAS INT8 tensor 要 k%4==0）。n=512 走 `cublasGemmEx`。
