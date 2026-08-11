# Binary sensor-set ablation

三個 RUN 使用同一份 binary windows 與 split checksum
ecba21a44e141523c692c3fbb092bb512b441de988408738f1ad5c9113562ea3。模型、nested calibration 與 threshold protocol 相同，只改變輸入 channels。下表是 per-held-out-subject mean±std。

| sensor set | channels / features | macro-F1 | fall recall | precision | AUPRC | false-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| waist accel+gyro | 6 / 66 | 0.9031±0.0518 | 0.9209±0.0672 | 0.7393±0.1428 | 0.8814±0.1013 | 0.01056±0.00630 |
| five wearable IMUs | 35 / 385 | 0.8951±0.0919 | 0.8961±0.2213 | 0.7773±0.1388 | 0.9194±0.0886 | 0.00872±0.00543 |
| wearable IMUs + EEG/infrared | 42 / 462 | 0.9009±0.0762 | 0.8989±0.2024 | 0.7854±0.1389 | 0.9157±0.0960 | 0.00835±0.00534 |

沒有單一 sensor set 在所有 headline metrics 同時佔優：腰部 6-axis 在事前固定 threshold protocol 下的 macro-F1 與 recall 最高，但 precision、AUPRC 與 false-positive rate 較差；35-channel wearable IMUs 的 AUPRC 最高；42-channel 版的 precision 與 false-positive rate 最好。未做配對顯著性檢定，不宣稱這些小差異有統計顯著。

- Waist core：reports/runs/EXP-XGB-BIN-CORE-IMU-001/results.json
- All wearable IMUs：reports/runs/EXP-XGB-BIN-ALL-IMU-001/results.json
- + ambient baseline：reports/runs/EXP-GBM-BIN-LOSO-001/results.json
