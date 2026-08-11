"""Benchmark torch-free ONNX end-to-end batch-1 latency on CPU / RTX 4090。"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import yaml
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.deployment.runtime import (  # noqa: E402
    create_onnx_session,
    predict_tcn_probability,
    predict_xgb_probability,
)
from fallsense.features import extract_handcrafted_features  # noqa: E402

FloatArray = NDArray[np.float64]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--config", type=Path, default=Path("configs/export.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/export"))
    return parser


def _calibrate(probability: FloatArray, metadata: dict[str, Any]) -> FloatArray:
    calibration = metadata["calibration"]
    score = float(calibration["coefficient"]) * probability + float(calibration["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(score, -709.0, 709.0)))


def _benchmark(function: Callable[[], object], warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        start = time.perf_counter_ns()
        function()
        samples[index] = (time.perf_counter_ns() - start) / 1_000_000.0
    return {
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(np.min(samples)),
    }


def _gpu_name() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "unavailable"


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    rows = cast(dict[str, dict[str, object]], payload["results"])
    lines = [
        "# ONNX batch-1 latency",
        "",
        f"- warmup: {payload['warmup']} iterations",
        f"- measured: {payload['iterations']} iterations",
        "- end-to-end includes NumPy preprocessing, ONNX inference and sigmoid calibration",
        "- XGBoost TreeEnsemble is measured on CPU; TCN is measured on CPU and RTX 4090 CUDA EP",
        "",
        "| model / provider | p50 ms | p95 ms | mean ms | min ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['p50_ms']:.4f} | {row['p95_ms']:.4f} | "
            f"{row['mean_ms']:.4f} | {row['min_ms']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    args = _parser().parse_args()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    benchmark = cast(dict[str, Any], config["benchmark"])
    warmup = int(benchmark["warmup"])
    iterations = int(benchmark["iterations"])
    window = np.asarray(np.load(args.windows)["values"][:1], dtype=np.float64)
    xgb_metadata = json.loads(
        (args.models / "xgb_binary/metadata.json").read_text(encoding="utf-8")
    )
    tcn_metadata = json.loads(
        (args.models / "tcn_binary/metadata.json").read_text(encoding="utf-8")
    )

    xgb_cpu = create_onnx_session(
        args.models / "xgb_binary/model.onnx",
        providers=("CPUExecutionProvider",),
    )
    tcn_cpu = create_onnx_session(
        args.models / "tcn_binary/model.onnx",
        providers=("CPUExecutionProvider",),
    )
    # ORT 會從 PyTorch wheel 目錄載入 CUDA/cuDNN DLLs，但 inference 本身不 import torch。
    ort.preload_dlls()
    tcn_gpu = create_onnx_session(
        args.models / "tcn_binary/model.onnx",
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )

    def xgb_inference() -> FloatArray:
        features = extract_handcrafted_features(window).values
        return _calibrate(predict_xgb_probability(xgb_cpu, features), xgb_metadata)

    def tcn_cpu_inference() -> FloatArray:
        return _calibrate(predict_tcn_probability(tcn_cpu, window, tcn_metadata), tcn_metadata)

    def tcn_gpu_inference() -> FloatArray:
        return _calibrate(predict_tcn_probability(tcn_gpu, window, tcn_metadata), tcn_metadata)

    results = {
        "XGBoost / CPU": _benchmark(xgb_inference, warmup, iterations),
        "TCN / CPU": _benchmark(tcn_cpu_inference, warmup, iterations),
        "TCN / CUDA": _benchmark(tcn_gpu_inference, warmup, iterations),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "batch_size": 1,
        "warmup": warmup,
        "iterations": iterations,
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "gpu": _gpu_name(),
        "available_providers": ort.get_available_providers(),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latency.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_markdown(payload, args.output_dir / "latency.md")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
