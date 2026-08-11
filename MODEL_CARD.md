# FallSense sensor model card

Version：0.1.0（research prototype）

## Model summary

Primary research model 是 binary fall/non-fall XGBoost，輸入為 42-channel sensor window 的
462 個 handcrafted time/frequency features。Secondary model 是 channel-masked causal 1D TCN，
但因同 split LOSO 結果較差，不是推薦模型。

- Task：binary fall detection（activities 1–5 的 fall segment vs non-fall）。
- Window：50 samples，stride 25，center Tag，任一 Tag 20 的 window 丟棄。
- Input：5 組 wearable IMU/luminosity + EEG + 6 infrared channels；缺 channel 保留 NaN。
- Output：calibrated fall probability；本機 deployment candidate threshold `0.10`。
- Public artifact policy：不發行 checkpoint、native model 或 ONNX；只保留 aggregate parity/latency reports。

## Intended use

- 研究、教學、subject-wise 評估與本機模型匯出驗證。
- 比較 subject shift、threshold 與模型選擇。
- 不是醫療器材，不可作為單一照護或緊急求援保障，不可用於臨床決策。

## Training and evaluation protocol

Official UP-Fall per-trial sensor CSV 產生 10,905 windows，含 349 fall windows。每一個 outer
fold 留一 subject 測試，訓練內的 calibration、threshold 與 imbalance 決策只使用
outer-train subjects。Scaler/normalizer 也只 fit train fold。Split checksum：
`ecba21a44e141523c692c3fbb092bb512b441de988408738f1ad5c9113562ea3`。

Primary `EXP-GBM-BIN-LOSO-001` 的 per-held-out-subject mean±std：

| headline metric | XGBoost | secondary TCN |
|---|---:|---:|
| macro-F1 | 0.9009±0.0762 | 0.6763±0.1362 |
| fall recall | 0.8989±0.2024 | 0.5306±0.3180 |
| precision | 0.7854±0.1389 | 0.4804±0.3072 |
| AUPRC | 0.9157±0.0960 | 0.4709±0.2868 |
| false-positive rate | 0.0083±0.0053 | 0.0562±0.0914 |

完整 per-subject 表、threshold sweep 與 confusion matrix 在
`reports/runs/EXP-GBM-BIN-LOSO-001/`；TCN 對應 `EXP-TCN-BIN-LOSO-001`。表中是
fold-wise evaluation models，不是以全部 subjects 重 fit 後對同一資料評分。

## Calibration and deployment threshold

評估時，每個 outer fold 以 outer-train grouped OOF probabilities 各自 fit sigmoid calibration
並選 threshold。本機 deployment candidate 則在全部 17 subjects 上 fit model，calibration 用全 subjects
GroupKFold OOF probabilities（coefficient `8.2221202925`、intercept `-6.2794338275`），預設
threshold `0.10`。此 all-subject fit 是 deployment scope，不是新的 held-out 效能數字。

## Export validation and latency

Native/ONNX 在 17-subject replay 的 XGBoost probability maximum absolute difference 為
`4.17e-7`，threshold-based headline metric difference 為 0。Batch-1 end-to-end CPU p50 為
`2.5076ms`（warmup 50、measured 500 iterations）。這些對應 registry
`VAL-EXPORT-XGB-001` `RUN`，報告在 `reports/export/`。模型檔本身未公開，因為資料集檔案與
derived-weight 的發行權利尚未取得可保存的明確授權；這些數字不是可下載模型的承諾。

## Limitations and risks

- 資料只有 17 位健康年輕受試者在受控環境做模擬跌倒；不代表真實老人跌倒。
- 沒有外部 dataset、長時間 ADL、每小時誤警、臨床或 prospective validation。
- Subject 9 fall recall 只有 `0.1739`，顯示 threshold transfer 可能在新人群上失敗。
- Subjects 5/9 缺 right-pocket channels；mask/NaN handling 不會恢復遺失資訊。
- Window false-positive rate 不等於真實佈署的每小時誤警。連續 window 也不是獨立事件。
- Vision/fusion performance 沒有訓練，registry 為 `NOT RUN`。Privacy mode 預設鎖定 sensor-only。

更完整的失敗分析與限制見 `reports/error_analysis.md` 與 `reports/limitations.md`。
