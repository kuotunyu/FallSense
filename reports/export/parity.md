# ONNX held-out parity

ONNX-backed classifiers 重跑與已登錄 runs 相同的 outer LOSO / inner grouped calibration protocol。

| model | probability diff | threshold-metric diff | AUPRC diff | result |
|---|---:|---:|---:|---|
| XGBoost | 4.1723251e-07 | 0 | 2.220446e-16 | PASS |
| TCN | 2.9802322e-07 | 0 | 0.0023133559 | PASS |

Probability tolerance = 1e-5；threshold-metric tolerance = 1e-4；rank-sensitive AUPRC tolerance = 0.005。
TCN probability 近似 ties 在 ONNX float32 kernels 可能有極小排序變動，因此 AUPRC 單獨設定 tolerance，不與 threshold-based metrics 混用。
