# EXP-XGB-11C-LOSO-001 — 11-class LOSO report

- status: `RUN`
- git commit: `c6baf238119a4c281675d1dfad475181f1b3b463`
- split checksum: `aae844779e35e3dbcd12c6e6e7513e07e0e8db50094edebfffb3bc374bc7e72a`
- calibration: multinomial logistic regression fitted only on outer-train grouped OOF probabilities
- per-subject macro-F1 averages only classes present in that held-out subject; missing-support classes are reported as unavailable

## Headline metric (per-held-out-subject mean±std)

- macro-F1: **0.7708±0.0816**

## Per-subject

| subject | macro-F1 | classes with support |
|---:|---:|---:|
| 1 | 0.9035 | 11 |
| 2 | 0.8256 | 11 |
| 3 | 0.7484 | 11 |
| 4 | 0.7919 | 11 |
| 5 | 0.6680 | 11 |
| 6 | 0.7933 | 11 |
| 7 | 0.7246 | 10 |
| 8 | 0.8005 | 11 |
| 9 | 0.8814 | 11 |
| 10 | 0.8132 | 10 |
| 11 | 0.8164 | 11 |
| 12 | 0.7032 | 10 |
| 13 | 0.7314 | 11 |
| 14 | 0.5760 | 11 |
| 15 | 0.7192 | 11 |
| 16 | 0.7487 | 11 |
| 17 | 0.8584 | 11 |

## Per-class (mean±std over held-out subjects with support)

| class | subjects | precision | recall | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|
| activity_1 | 17 | 0.6407±0.2084 | 0.8510±0.2442 | 0.6709±0.1322 | 0.8722±0.1134 |
| activity_2 | 14 | 0.4946±0.3508 | 0.7214±0.3054 | 0.5408±0.3027 | 0.6967±0.2689 |
| activity_3 | 17 | 0.5192±0.2618 | 0.7190±0.2960 | 0.5760±0.2548 | 0.7290±0.2147 |
| activity_4 | 17 | 0.5263±0.2720 | 0.8941±0.2561 | 0.6275±0.2445 | 0.7443±0.1998 |
| activity_5 | 17 | 0.5293±0.2668 | 0.6566±0.3208 | 0.5500±0.2364 | 0.6966±0.1999 |
| activity_6 | 17 | 0.9972±0.0056 | 0.9812±0.0530 | 0.9883±0.0295 | 0.9986±0.0036 |
| activity_7 | 17 | 0.9395±0.1282 | 0.9618±0.0480 | 0.9457±0.0836 | 0.9776±0.0389 |
| activity_8 | 17 | 0.9275±0.1069 | 0.8721±0.2646 | 0.8812±0.2184 | 0.9591±0.0794 |
| activity_9 | 17 | 0.6120±0.2217 | 0.9289±0.1486 | 0.7210±0.2058 | 0.8518±0.1862 |
| activity_10 | 17 | 1.0000±0.0000 | 0.9971±0.0088 | 0.9985±0.0044 | 1.0000±0.0000 |
| activity_11 | 17 | 0.9729±0.0619 | 0.9187±0.0583 | 0.9427±0.0403 | 0.9778±0.0372 |
