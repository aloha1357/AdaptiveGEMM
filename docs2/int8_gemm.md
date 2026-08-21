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
| AdaptiveGEMM Phase 22 | 13.6 ms | 1262 | 6.0× | 3.40e-9 |
| cuBLAS 13 Ozaki emu | 8.5 ms | 2014 | 9.6× | 2.84e-13 |

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

公平賽之後要補：把 cuBLAS `MaxMantissaBitCount` 鎖在 16 / 24 / 34 / 53，跟 Phase 22 畫在同一張時間–誤差圖上。現在拿 3e-9 去比人家的 3e-13，沒意義。

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

Bank conflict 沒有一次打掉。下一步：下一個質數的 k=0 提前 prefetch，或公平賽 mantissa bits。

---

## 什麼叫這輪成功

- 誤差不能壞：2048 max error 仍在 `1e-8` 以內，理想維持 `~3e-9`。
- Phase 22 wall time 往 10 ms 靠近。到 8.5 ms 才是「跟預設 emu 同一量級」，那是 Day 10 的天花板，不是今天。
- Demo 預設維持 Phase 22。README 的啟動方式不准寫回 Phase 24。
