# Current limitations

- 資料來自受控實驗室環境的 17 位 subjects，不代表長照機構、居家、戶外或臨床人群。
- 未做外部 dataset 驗證，所有結論只限 UP-Fall sensor protocol 下的未見 subject generalization。
- Binary fall windows 只有 349/10,905，11-class 的稀少類別只有 58–83 windows；即使用 train-fold-only weights，結果仍對少數事件很敏感。
- 50-sample windows 以 25-sample stride 重疊，所以 window 不是獨立事件；本專案以 subject 為 split group 防止跨 fold 洩漏，但不把 window 數解釋為獨立樣本數。
- Tag 20 缺少可靠語義，primary configs 會丟棄任何含 Tag 20 的 window；這使 subjects 7/10/12 的 11-class Activity 2 無 support。
- Subjects 5/9 缺整組 right-pocket channels，Subject 2 Activity 5 缺 EEG。masking 可防止虛假數值貢獻，但不會恢復丟失的資訊。
- 目前 false-positive rate 是以實驗 windows 計算，不是「每小時誤警數」；在沒有長時間連續 ADL 測試前，不能外推為真實部署誤警負擔。
- Calibration 與 threshold 只在 outer-train grouped OOF data 上選擇，但 Subject 9 顯示 threshold 轉移仍可能造成高 ranking quality 卻低 recall。
- 公開 release 不含 derived model weights 或 model-backed demo；本機 deployment prototype
  也不是真正即時 sensor stream，且沒有長時間佈署的每小時誤警評估。
- vision/late-fusion 訓練為 `NOT RUN`；只交付 privacy-first 介面、對齊拒絕測試與
  resource-gated Colab notebook。即使未來加入 vision，房間隱私、遮擋、夜間低照度與攝影機
  domain shift 仍是主要風險。
- 沒有外部資料或臨床驗證；這不是醫療器材，也不應作為單一安全保障。
