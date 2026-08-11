# EXP-QUICK-XGB-001 — binary LOSO report

- status: `QUICK-MODE-ONLY`
- git commit: `62e345b28495cfba02fb4f95d8270cd0ea375a45`
- split checksum: `9b30519d405b0ed23843b6e3a1c0dc20b8910d4a7bf6da8a52cf76426fc44ba1`
- threshold: 每個 outer fold 只以 outer-train OOF calibrated probabilities 選擇

## Headline metrics（per-held-out-subject mean±std）

| metric | mean±std |
|---|---:|
| macro-F1 | 1.0000±0.0000 |
| fall recall | 1.0000±0.0000 |
| precision | 1.0000±0.0000 |
| AUPRC | 1.0000±0.0000 |
| false-positive rate | 0.0000±0.0000 |

## Per-subject

| subject | threshold | macro-F1 | fall recall | precision | AUPRC | FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 2 | 0.05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 3 | 0.05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
