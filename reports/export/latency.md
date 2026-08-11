# ONNX batch-1 latency

- warmup: 50 iterations
- measured: 500 iterations
- end-to-end includes NumPy preprocessing, ONNX inference and sigmoid calibration
- XGBoost TreeEnsemble is measured on CPU; TCN is measured on CPU and RTX 4090 CUDA EP

| model / provider | p50 ms | p95 ms | mean ms | min ms |
|---|---:|---:|---:|---:|
| XGBoost / CPU | 2.5076 | 3.6962 | 2.7833 | 2.4495 |
| TCN / CPU | 0.1546 | 0.1617 | 0.1669 | 0.1328 |
| TCN / CUDA | 0.5584 | 1.1243 | 0.6004 | 0.4991 |
