# Binary LOSO：XGBoost 與 TCN 同 split 對比

兩個 `RUN` 都使用 split checksum
`ecba21a44e141523c692c3fbb092bb512b441de988408738f1ad5c9113562ea3`，並以每個 outer-train fold 的 grouped OOF probabilities 進行 sigmoid calibration 與 threshold selection。下表是 per-held-out-subject mean±std，不是 pooled score。

| model | macro-F1 | fall recall | precision | AUPRC | false-positive rate |
|---|---:|---:|---:|---:|---:|
| XGBoost handcrafted | 0.9009±0.0762 | 0.8989±0.2024 | 0.7854±0.1389 | 0.9157±0.0960 | 0.00835±0.00534 |
| channel-masked TCN | 0.6763±0.1362 | 0.5306±0.3180 | 0.4804±0.3072 | 0.4709±0.2868 | 0.05617±0.09136 |

結論：在這個事前固定的小型 TCN config 下，handcrafted XGBoost 在五個 headline metrics 上都明顯較好，因此仍是 binary best model。TCN 結果保留為負結果；不使用 held-out subjects 表現反覆調參。

- XGBoost：`reports/runs/EXP-GBM-BIN-LOSO-001/results.json`
- TCN：`reports/runs/EXP-TCN-BIN-LOSO-001/results.json`
