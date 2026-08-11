# FallSense data card

## Dataset and provenance

FallSense 使用官方 **UP-Fall Detection Dataset / HAR-UP** per-trial sensor CSV。資料由
17 位健康年輕受試者在受控環境執行 5 種模擬跌倒與 6 種 ADL，每個 activity
最多 3 trials。必須引用：Martínez-Villaseñor et al., *Sensors* 2019, 19(9),
1988，DOI `10.3390/s19091988`。

官方網站：https://sites.google.com/up.edu.mx/har-up/

## Repository scope

- Repository 不包含、鏡像或重新散布任何真實受試者資料。
- Phase 1 downloader 只允許 `SubjectXActivityYTrialZ.csv` 與 `Tagged_TimeStamps.zip`。
- `*Camera*.zip` 永久排除；完整多模態資料超過 850GB，不是 sensor MVP 依賴。
- `tests/fixtures/` 的 CSV 全部由固定 seed 程式生成，不是真實人體訊號。
- 公開 release 不包含由真實資料衍生的 checkpoints、native models 或 ONNX。

## Verified local profile

- 官方 listing：17 subjects × 11 activities × 3 trials = 561 sensor CSV。
- `Subject8Activity11Trial{2,3}.csv` 只有 header，無資料列；usable trials = 559。
- 294,678 sensor rows；per-trial median 192，range 140–1,145。
- Timestamp 估測 sampling rate median 20.249Hz，range 15.284–21.220Hz。
- 42 sensor channels：5 組 wearable IMU/luminosity（3-axis accel + 3-axis gyro + light）、
  EEG 1ch、infrared 6ch。加 timestamp / Subject / Activity / Trial / Tag 後標準化為 47 columns。
- Raw sensor CSV 為 82,743,900 bytes；加 timestamps 共 84,521,289 bytes。
- Binary 50/25 center-label artifact：10,905 windows，349 fall / 10,556 non-fall。

上述數字對應 M1/M2 本機驗證與 manifest，不是模型效能。

## Labels and preprocessing

- Row-level Tag 主值為 1–11；1–5 是 fall segments，6–11 是 ADL。
- Tag 20 缺乏可靠語義。DEC-003 規定 primary binary config 丟棄任一含 Tag 20 的 window，
  不擅自映射為 non-fall。
- Window 長 50 samples、stride 25、以 center row Tag 標記。
- Subjects 5/9 全部缺 right-pocket 7 channels；Subject 2 Activity 5 缺 EEG。
  XGBoost 保留 NaN；TCN 使用顯式 channel mask。
- Scaler/normalizer、calibration、threshold 與 class imbalance 決策只在 train folds 內完成。

## Splits and integrity

Primary evaluation 以 subject 為 group 做 leave-one-subject-out，絕不 random row/frame split。
Binary split checksum：
`ecba21a44e141523c692c3fbb092bb512b441de988408738f1ad5c9113562ea3`。
11-class split checksum：
`aae844779e35e3dbcd12c6e6e7513e07e0e8db50094edebfffb3bc374bc7e72a`。

Download manifest 記錄 remote file ID、相對路徑、bytes 與 SHA-256；known-unusable、下載錯誤與
checksum mismatch 分開報告。No-subject-overlap 與 repo-data guard 都是自動測試。

## License, privacy, and limitations

Dataset page 提供公開存取與引用論文，但本專案沒有找到一份明確適用於
dataset files 的 open-data license。論文的 CC BY 4.0 不會被延伸解讀為資料檔授權。
因此資料必須由使用者從官方來源自行取得，並只留在 gitignored `data/`。

受試者是健康年輕人、跌倒是模擬、場景是實驗室；不代表長照、居家、夜間、戶外或
臨床人群。真實世界 fall prevalence 也遠低於實驗資料。實驗 AUPRC、threshold 與
window FPR 不可直接當成臨床效能或每小時誤警。
