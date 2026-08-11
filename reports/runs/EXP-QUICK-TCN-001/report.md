# EXP-QUICK-TCN-001 — binary LOSO report

- status: `QUICK-MODE-ONLY`
- git commit: `228533faf23ce2ebc5525cc84b357dc550653772`
- split checksum: `9b30519d405b0ed23843b6e3a1c0dc20b8910d4a7bf6da8a52cf76426fc44ba1`
- threshold: 每個 outer fold 只以 outer-train OOF calibrated probabilities 選擇

## Headline metrics（per-held-out-subject mean±std）

| metric | mean±std |
|---|---:|
| macro-F1 | 0.6562±0.2978 |
| fall recall | 0.3333±0.5774 |
| precision | 0.3333±0.5774 |
| AUPRC | 0.3705±0.5452 |
| false-positive rate | 0.0000±0.0000 |

## Per-subject

| subject | threshold | macro-F1 | fall recall | precision | AUPRC | FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 2 | 0.15 | 0.4857 | 0.0000 | 0.0000 | 0.0506 | 0.0000 |
| 3 | 0.10 | 0.4828 | 0.0000 | 0.0000 | 0.0608 | 0.0000 |
