# 為什麼 34-bit 我們還輸 cuBLAS 2.8×

用 FlashAttention / Triton 的 IO 觀點，不是「再 unroll 一次」。

cuBLAS 的 Ozaki **原始碼不公開**。下面對照的是 NVIDIA 文件寫死的 Scheme I、
開源 ozIMMU（NVIDIA 引用的那篇 INT8 DGEMM 論文實作）、以及 EmuGEMM 把
FlashAttention 那套融合講清楚的版本。我們輸在同一件事：**中間結果一直進 HBM，
而且演算法選了比較貴的那條。**

---

## 一句話

FlashAttention 贏在「softmax 的 S、P 不要寫回 HBM」。
Triton 贏在「一個 kernel 把 load–MMA–store 融掉，compiler 幫你 pipeline」。

cuBLAS 13 FP64 emu 是 **Ozaki Scheme I：8-bit slice，INT8×INT8→INT32 是精確的，
用 shift-add 拼回來**。成本跟 slice 數平方，但每次乘法是一顆量產 INT8 GEMM。

我們是 **2-slice Ozaki（hi/lo）再套 7 質數 CRT（Scheme II）**：4 × 7 = 28 次
HBM 上的 INT8 K-sweep，還外加 64-bit Barrett。同一 34-bit 預算，他們大約
5 slice × 三角配對 ~15 次 INT8 GEMM，而且 INT8 kernel 吃到 ~80% 峰值。

實測 2048³：cuBLAS 34-bit **3.97 ms / 7.6e-10**，Phase 22 **~11 ms / 3.4e-9**。

---

## cuBLAS 實際在做什麼（文件 + 開源對照）

NVIDIA 自己寫（CUDA 13 cuBLAS *Floating Point Emulation / Fixed-Point*）：

1. 同一列 A / 同一欄 B 共用一個 2 的次方 scale，把值邏輯縮到 [-1, 1]
2. 符號 + mantissa 塞進 **8-bit integer slices**
3. `sliceCount = ceildiv(mantissaBitCount + 1, 8)`，**計算量跟 slice 數平方成長**
4. Dynamic mantissa（ADP）決定用幾 bit；太多就退回 native FP64
5. Workspace 最多 8 GB，用來放固定點中間結果

這就是 **Scheme I**。不是 CRT。

開源 ozIMMU（Ootomo / Ozaki / Yokota，cuBLAS 引用的那條 INT8 DGEMM）步驟幾乎
是同一張圖，原始碼在 `gemm.cu`：

```c
// 1) 切成 num_split 張 INT8
split_AB_int8<double>(..., num_split, bits_per_int8);

// 2) 對每一對 (A_id, B_id) 呼叫量產 INT8 GEMM
for (const auto &gemm_pair_config : gemm_pair_config_list) {
    cublasGemmEx(..., CUDA_R_8I, CUDA_R_8I, CUDA_R_32I,
                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    accumulate_in_f64(c_f64_ptr, c_i32_ptr, ...,
                      bits_per_int8 * (A_id + B_id - 2) - ...);
}
axby(...);  // 乘回 per-row / per-col scale
```

重點：

- 切完之後，**乘法外包給 `cublasGemmEx` INT8 Tensor Core**（Ada 上那顆 ~80% 峰值的 kernel）
- 重建是 `int32 << 位移` 累加進 double，**沒有 7 質數、沒有 `% M`**
- 這還是「naive Scheme I」：每一對 slice 一個 kernel，INT32 部分積寫回 HBM

cuBLAS 產品化之後，EmuGEMM（2026）量到：Hopper 上 cuBLAS emu 只在 slice 少
（p≤5）時贏 native FP64 TC；**中間 INT32 還是會進 GMEM**。他們把 FlashAttention
那套搬過來：所有 `p(p+1)/2` 個乘積融進一顆 kernel，slice 只從 HBM 讀一次，
accumulator 留在 RF / TMEM，最後 shift-reduce 寫一次 C。

HBM 流量（EmuGEMM 的式子）：

```
naive:  O(p²) * (M+N)K   +  O(p²) * 4 MN   +  8 MN
fused:  O(p)  * (M+N)K   +  8 MN
```

p=5（34-bit）時，光 operand 就是 15 次對 5 次。INT32 部分積再乘 4 倍元素大小。

**我們看不到 cuBLAS.dll 裡的 PTX。** 但 34-bit 3.97 ms 這個數字，已經比
「15 次手寫 30% 峰值 GEMM」快太多，也比 ozIMMU 那種每對 slice 各 launch 一次
更像「至少用了量產 INT8 GEMM，而且 packing / 重建夠乾」。Ada 沒有 Hopper TMA /
wgmma，cuBLAS 在 4060 上比較可能是：**切 slice → 一串高度優化的 INT8 GEMM
（可能有 stream overlap）→ shift-add**，還不到 EmuGEMM 那種全融。就算如此，
單次 INT8 GEMM 品質已經夠把我們甩開。

---

## 我們實際在跑的步驟

`execute()` Phase 22：

```text
1. memset C
2. precompute_modulo_hl_A/B   FP64 → 7 質數 × {hi, lo} INT8，寫進 HBM
3. 4 次 crt_fast_kernel
      pass 0: Ah × Bh   × 2^{-34}
      pass 1: Ah × Bl   × 2^{-51}
      pass 2: Al × Bh   × 2^{-51}
      pass 3: Al × Bl   × 2^{-68}
```

每一顆 `crt_fast_kernel`（`AdaptiveOzaki.cu`）：

```cuda
for (int p = 0; p < 7; p++) {          // 質數序列，每次重新掃完整個 K
    // cp.async 從 HBM 載入這一張餘數矩陣的 128×32 / 64×32 tile
    for (int ck = 0; ck < k; ck += 32) {
        mma.sync.m16n8k32.s8          // 這才是 Tensor Core
    }
    v = cf % p;                        // 現在是 magic
    acc = (acc + v * f_p) % M;         // Barrett，但仍是 64-bit
}
C[i,j] += reconstruct(acc) * inv;      // 4 個 pass 各寫一次 C
```

對照 FlashAttention：我們**有**把 CRT 拼進同一顆 kernel（沒把 7 張 INT32 C 寫回），
這是對的。但：

| | FlashAttention / 融好的 Scheme I | 我們 |
|---|---|---|
| 中間矩陣 | S、P 留在 SRAM | 14 張 INT8 餘數（7×2）先寫 HBM，再讀 4 遍 |
| K 迴圈 | 每個 Q tile 沿 K 走一次 | **每個質數沿 K 走一次 × 7 × 4 pass** |
| 乘法本體 | 一顆吃滿的 MMA | Ada `mma.sync` 128×64，~30% 峰值 |
| 重建 | shift（2 的次方） | 7 次 Barrett + 4 次 FP64 `C +=` |
| launch | 1 | pack + 4 CRT |

2048³ 粗帳（只算 operand INT8 從 HBM 再讀）：

- 我們：28 次「完整 K 的 A/B tile stream」
- Scheme I p=5 若融合：5 次
- Scheme I p=5 若不融合：15 次，但每次是 cuBLAS INT8

這就是 2.8× 從哪裡來：大約一半演算法（28 vs 15、CRT vs shift），一半 kernel
品質（40 TOPS vs ~100 TOPS）。

---

## 為什麼「再優化 mma.sync」追不回

Barrett、XOR swizzle 已經證明：除法跟一點 bank conflict 能從 13.6 收到 10.3 ms。
再往下是：

- 7 個質數 **序列** 重讀 K（FlashAttention 禁止的那種 IO）
- 4 個 slice 乘積 **4 顆 kernel**，C 四次 RMW
- tile 太小、沒有 TMA、沒有把 5 張 8-bit slice 交錯排進同一條 K
- 17-bit slice 的乘積 ×K 塞不進 int32，所以 **被迫** CRT

cuBLAS 選 8-bit 就是為了讓「一次 MMA = 一次精確整數 GEMM」。
我們選 17-bit + CRT 是為了少 slice，結果 slice 變少、**每次變肥**，HBM 次數反而更多。

---

## 接下來如果要追

跟 FlashAttention 同一句話：**中間張量不要進 HBM，K 只掃一次。**

具體就是 Scheme I：

1. 切 5 張 8-bit（34-bit）
2. 一顆 kernel（或至少呼叫 cuBLAS INT8）做完所有 pair
3. register 裡 shift-add 寫 C

那才是跟 3.97 ms 同一條路。繼續調 `crt_fast_kernel` 的 `mma.sync` 是在優化一條
比較貴的 Scheme II。
