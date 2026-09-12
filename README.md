# FallSense

[![CI](https://github.com/kuotunyu/FallSense/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/FallSense/actions/workflows/ci.yml)

隱私優先的穿戴式感測器跌倒偵測研究管線。專案以 UP-Fall / HAR-UP 為研究案例，
重點不是做出一個漂亮但不可稽核的分類 Demo，而是把資料治理、subject-wise 評估、
calibration、threshold selection、錯誤分析與模型匯出放在同一條可追溯流程中。

## 這個 Repository 證明什麼

- 以 subject 為 group 的 outer LOSO 與 inner grouped folds，避免同一受試者跨 train/test。
- scaler、class imbalance、calibration 與 threshold 全部只在 training folds 決定。
- XGBoost handcrafted-feature baseline、channel-masked causal TCN 與 11-class extension。
- 固定 seed 的合成 fixture，可在沒有真實受試者資料時驗證 parsing、windowing、split 與 reporting。
- 結果 registry、per-subject 報告、threshold sweep、confusion matrix、ONNX parity 與 latency 證據。
- GitHub CI 與 repository guard，阻止真實資料、衍生模型、秘密檔和內部 session 文件誤入 Git。

## 主要結果

Primary binary experiment `EXP-GBM-BIN-LOSO-001` 使用 17 位受試者的 held-out-subject
folds；以下為 per-subject mean±std，不是 pooled-window 分數：

| metric | XGBoost | TCN |
|---|---:|---:|
| macro-F1 | 0.9009±0.0762 | 0.6763±0.1362 |
| fall recall | 0.8989±0.2024 | 0.5306±0.3180 |
| precision | 0.7854±0.1389 | 0.4804±0.3072 |
| AUPRC | 0.9157±0.0960 | 0.4709±0.2868 |
| false-positive rate | 0.0083±0.0053 | 0.0562±0.0914 |

完整證據位於 [`reports/`](reports/)。這些結果只代表受控實驗室中的健康年輕受試者，
不能外推成長照、居家或臨床效能；Subject 9 的 fall recall 只有 `0.1739`，是重要的失敗案例。

## Clean-clone 驗證

不需要 UP-Fall 真實資料、GPU、PyTorch、模型權重或 API key：

```powershell
git clone https://github.com/kuotunyu/FallSense.git
cd FallSense
uv python install 3.10
uv sync --python 3.10 --frozen --extra dev --extra download --extra plot
uv run python scripts/check_repo_guard.py
uv run ruff check .
uv run mypy
uv run pytest
```

重新產生合成 fixture：

```powershell
uv run python scripts/generate_synthetic_fixture.py --output tests/fixtures/upfall_synthetic
```

## 真實資料重現邊界

Repository 不包含、鏡像或重新散布 UP-Fall 真實受試者資料。資料集頁面提供公開存取與
引用要求，但本專案沒有找到明確適用於 dataset files 的 open-data license。使用者必須從
官方來源自行取得資料，並只放在 gitignored 的 `data/`。

```powershell
uv run python scripts/data_download.py --help
uv run python scripts/make_windows.py --raw-dir data/raw/upfall --config configs/windowing_50_25.yaml --output-dir data/processed/real_50_25_center
```

XGBoost training 需額外安裝 `train` extra；ONNX 匯出需再加 `export`。TCN config 固定記錄
PyTorch/CUDA runtime，請依 `configs/tcn_binary.yaml` 的版本與官方 index 建立環境，不要以
不相同 runtime 重新標示既有結果。

## 為什麼沒有公開模型與線上 Demo

先前本機版本完成過 ONNX parity、容器與 UI smoke test，但模型是由 UP-Fall 資料衍生。
在 dataset-file 與 derived-weight 權利沒有取得可保存的明確授權前，本公開版刻意不發行
checkpoint、native model、ONNX 或 model-backed Space。`reports/export/` 只保留聚合的 parity
與 latency 證據；這不是缺檔，而是 release policy。

若未來取得授權，模型發布必須另做 hash-locked release、clean-container smoke、license inventory
與公開 artifact provenance，不能直接把本機 `artifacts/models/` 加進 Git。

## 文件

- [`DATA_CARD.md`](DATA_CARD.md)：資料來源、切分、缺失與授權界線。
- [`MODEL_CARD.md`](MODEL_CARD.md)：模型、結果、限制與 artifact policy。
- [`docs/architecture.md`](docs/architecture.md)：資料到研究證據的系統邊界。
- [`reports/error_analysis.md`](reports/error_analysis.md)：失敗案例與 subject shift。
- [`reports/limitations.md`](reports/limitations.md)：不可宣稱的事項。

FallSense 是研究原型，不是醫療器材，也不能作為單一照護或緊急求援保障。
