# EXP-TCN-11C-LOSO-001 — 11-class LOSO report

- status: `RUN`
- git commit: `c6baf238119a4c281675d1dfad475181f1b3b463`
- split checksum: `aae844779e35e3dbcd12c6e6e7513e07e0e8db50094edebfffb3bc374bc7e72a`
- calibration: multinomial logistic regression fitted only on outer-train grouped OOF probabilities
- per-subject macro-F1 averages only classes present in that held-out subject; missing-support classes are reported as unavailable

## Headline metric (per-held-out-subject mean±std)

- macro-F1: **0.4024±0.0656**

## Per-subject

| subject | macro-F1 | classes with support |
|---:|---:|---:|
| 1 | 0.4936 | 11 |
| 2 | 0.4198 | 11 |
| 3 | 0.4698 | 11 |
| 4 | 0.4778 | 11 |
| 5 | 0.3477 | 11 |
| 6 | 0.3415 | 11 |
| 7 | 0.3660 | 10 |
| 8 | 0.4815 | 11 |
| 9 | 0.4713 | 11 |
| 10 | 0.3388 | 10 |
| 11 | 0.3079 | 11 |
| 12 | 0.4459 | 10 |
| 13 | 0.3786 | 11 |
| 14 | 0.3397 | 11 |
| 15 | 0.4327 | 11 |
| 16 | 0.3013 | 11 |
| 17 | 0.4261 | 11 |

## Per-class (mean±std over held-out subjects with support)

| class | subjects | precision | recall | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|
| activity_1 | 17 | 0.0831±0.1425 | 0.3578±0.4091 | 0.1086±0.1661 | 0.2925±0.2841 |
| activity_2 | 14 | 0.1973±0.2886 | 0.4452±0.3722 | 0.2103±0.2324 | 0.3754±0.3807 |
| activity_3 | 17 | 0.1882±0.2516 | 0.2919±0.3116 | 0.1921±0.1855 | 0.3298±0.2288 |
| activity_4 | 17 | 0.1049±0.1570 | 0.3686±0.4358 | 0.1460±0.2050 | 0.3104±0.3616 |
| activity_5 | 17 | 0.1428±0.1960 | 0.3192±0.3658 | 0.1633±0.1876 | 0.3154±0.2538 |
| activity_6 | 17 | 0.8837±0.2384 | 0.6171±0.2542 | 0.7075±0.2453 | 0.9070±0.1278 |
| activity_7 | 17 | 0.6612±0.3726 | 0.6338±0.3744 | 0.6214±0.3418 | 0.8121±0.2050 |
| activity_8 | 17 | 0.5648±0.4173 | 0.5659±0.4369 | 0.5393±0.3978 | 0.7764±0.3034 |
| activity_9 | 17 | 0.1108±0.1517 | 0.4103±0.4024 | 0.1405±0.1696 | 0.2256±0.1967 |
| activity_10 | 17 | 0.7663±0.2586 | 0.8137±0.3126 | 0.7605±0.2643 | 0.9043±0.2225 |
| activity_11 | 17 | 0.9277±0.1168 | 0.7629±0.2345 | 0.8059±0.1784 | 0.9471±0.0384 |
