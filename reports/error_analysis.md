# Error analysis

本文只分析已登錄的 RUN 結果，不反向用 held-out subject 表現調整已發佈 config。

## Binary XGBoost subject outliers

- Subject 9 的 fall recall 只有 0.1739，但 precision 為 1.0000、AUPRC 為 0.9572、false-positive rate 為 0.0000。這表示該 subject 的 ranking 本身仍強，主要失敗發生在 outer-train 選定 threshold 轉移到該 subject 後過於保守；這是依 results 進行的推論，不是已驗證因果。
- Subject 14 的 AUPRC 0.6155 是 binary baseline 主要 ranking outlier，同時 false-positive rate 0.0144。這比 Subject 9 更像整體類別可分性下降。
- Subjects 5 與 9 的 right-pocket 7 channels 整個缺失；Subject 5 仍有 fall recall 0.9375，因此不能單憑兩個 subjects 就宣稱 missing pocket 是 Subject 9 失敗的原因。

本機曾以原始 sensor/Tag 對齊圖檢查 Subject 9 Activity 1 Trial 1，未發現明顯的 parser 或
label 時間錯位。由於該圖包含真實受試者衍生 signal trace，公開 release 不附圖；這項檢查
只排除已觀察到的時間錯位，不能證明 threshold transfer failure 的因果。

## 11-class confusion patterns

XGBoost 11-class 的總 confusion matrix 只用來找模式；headline 仍是 per-subject macro-F1。最大的 row-normalized off-diagonal patterns 為：

| true activity | predicted activity | count | row proportion |
|---:|---:|---:|---:|
| 2 | 1 | 10 | 0.1724 |
| 3 | 5 | 13 | 0.1667 |
| 5 | 4 | 13 | 0.1566 |
| 5 | 3 | 12 | 0.1446 |
| 1 | 2 | 8 | 0.1176 |
| 3 | 4 | 8 | 0.1026 |
| 8 | 7 | 178 | 0.0854 |

稀少 fall activities 1–5 之間互混是最明顯的問題，而 support 較多的 Activity 8 也常被判為 7。每類 per-subject F1/AUPRC 完整數字在 reports/runs/EXP-XGB-11C-LOSO-001/report.md；subjects 7/10/12 在 drop tag20 後沒有 Activity 2 support，報告不把無法評估的類別當成 0 分。

## 尚未能回答

現有 results.json 保留 fold-level confusion 與 probabilities 衍生指標，但未保留每個 window 的 prediction artifact，因此本次不捏造「某個特定 window 被誤判」的說法。後續若要做 waveform-level attribution，應先於新的事前登錄 experiment 中加入 window-level prediction export。
