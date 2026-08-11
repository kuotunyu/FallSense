"""以 raw sensor windows 訓練 channel-masked TCN nested LOSO baseline。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.data.schema import SENSOR_COLUMNS  # noqa: E402
from fallsense.evaluation.harness import (  # noqa: E402
    GBMConfig,
    ModelFactory,
    aggregate_subject_results,
    evaluate_loso,
    subject_results_payload,
)
from fallsense.evaluation.reporting import plot_report_figures, write_markdown_report  # noqa: E402
from fallsense.evaluation.results import (  # noqa: E402
    build_results_payload,
    save_results,
    sha256_path,
)
from fallsense.evaluation.splits import make_loso  # noqa: E402
from fallsense.models.tcn import TCNClassifier, TCNConfig  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/tcn_binary.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("RUN", "QUICK-MODE-ONLY"), required=True)
    return parser


def _model_factory(config: TCNConfig, input_channels: int) -> ModelFactory:
    def factory(_labels: np.ndarray[Any, np.dtype[np.int64]], seed: int) -> TCNClassifier:
        # pos_weight 在 classifier.fit 內從當次收到的 train fold labels 計算。
        return TCNClassifier(input_channels, config, seed=seed)

    return factory


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_payload = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    model_config = TCNConfig(**cast(dict[str, Any], config_payload["model"]))
    evaluation_payload = cast(dict[str, Any], config_payload["evaluation"])
    evaluation_config = GBMConfig(
        random_state=int(evaluation_payload["random_state"]),
        inner_folds=int(evaluation_payload["inner_folds"]),
        calibration=str(evaluation_payload["calibration"]),
    )
    archive = np.load(args.windows)
    windows = np.asarray(archive["values"], dtype=np.float64)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    groups = np.asarray(archive["groups"], dtype=np.int64)
    stored_mask = np.asarray(archive["channel_mask"], dtype=np.bool_)
    derived_mask = np.any(np.isfinite(windows), axis=1)
    if not np.array_equal(stored_mask, derived_mask):
        raise ValueError("windows.npz channel_mask 與 values 的 NaN pattern 不一致")
    if windows.ndim != 3 or windows.shape[2] != len(SENSOR_COLUMNS):
        expected_shape = f"[N,T,{len(SENSOR_COLUMNS)}]"
        raise ValueError(f"預期 raw windows shape {expected_shape}，實際為 {windows.shape}")
    expected_torch = str(cast(dict[str, Any], config_payload["runtime"])["torch"])
    if torch.__version__ != expected_torch:
        raise RuntimeError(f"config 要求 torch {expected_torch}，實際為 {torch.__version__}")
    split_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    folds = make_loso(groups)
    print(
        json.dumps(
            {
                "device": model_config.device,
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "windows": len(windows),
                "subjects": len(folds),
                "models_to_fit": len(folds) * (evaluation_config.inner_folds + 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results = evaluate_loso(
        windows,
        labels,
        groups,
        folds,
        config=evaluation_config,
        model_factory=_model_factory(model_config, windows.shape[2]),
    )
    channel_hash = hashlib.sha256("\n".join(SENSOR_COLUMNS).encode()).hexdigest()
    payload = build_results_payload(
        run_id=args.run_id,
        status=args.status,
        model="TCN-raw-window-channel-mask",
        task="binary-fall-detection",
        root=Path.cwd(),
        config=config_payload,
        config_hash=sha256_path(args.config),
        split_checksum=str(split_payload["checksum"]),
        data_manifest_checksum=sha256_path(args.data_manifest),
        feature_count=len(SENSOR_COLUMNS),
        feature_hash=channel_hash,
        per_subject=subject_results_payload(results),
        aggregate=aggregate_subject_results(results),
    )
    payload["runtime"] = {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    payload["imbalance"] = {
        "loss": "BCEWithLogitsLoss",
        "weight": "negative_count / positive_count",
        "scope": "each inner/final train fold only",
    }
    save_results(payload, args.output_dir / "results.json")
    write_markdown_report(payload, args.output_dir / "report.md")
    plot_report_figures(payload, args.output_dir)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
