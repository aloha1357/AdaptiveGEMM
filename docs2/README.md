# Docs 2｜INT8 GEMM 優化開發記錄

這份跟 `docs/` 裡 Phase 24/26 的舊 NCU 報告分開。

`docs/` 記的是「CRT kernel 曾經 36% tensor pipe、Phase 24 整條 130 ms」。
`docs2/` 記的是現在這條線：**Phase 22 已經贏 native DGEMM，接下來要吃滿 INT8 Tensor Core，才有機會靠近 cuBLAS 13 Ozaki emu。**

對應 iThome 系列 **Day 8、Day 9、Day 10**。Blog 正文還沒收進來，先把工程上要做什麼寫在這裡。

| 檔 | 內容 |
|---|---|
| [int8_gemm.md](int8_gemm.md) | 目標、帳、三天怎麼切、kernel 要改什麼、每次重測的數字 |

舊報告還在：

- `docs/31_Phase24_26_Performance_Tuning_and_Occupancy.md`
- `README.md`（對外說明：贏 native、輸 cuBLAS emu、預設 Phase 22）
