# Reports

本目錄只保存可追溯的研究證據：正式 experiment ID、config/split/data checksums、
per-subject aggregate metrics、confusion matrices、threshold sweeps、error analysis 與
privacy-safe export summaries。

不得加入真實 windows、row-level predictions、原始 sensor traces、checkpoint、native model
或 ONNX。公開 headline 數字必須能回溯到 `reports/runs/<experiment-id>/results.json`，且應優先
報告 held-out-subject mean±std，不以 pooled-window accuracy 取代。
