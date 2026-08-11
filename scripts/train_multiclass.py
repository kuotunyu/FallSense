"""訓練與 binary 嚴格分離的 11-class sensor LOSO model。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.data.schema import SENSOR_COLUMNS  # noqa: E402
from fallsense.evaluation.multiclass import (  # noqa: E402
    MulticlassConfig,
    MulticlassGBMConfig,
    MulticlassModelFactory,
    aggregate_multiclass_results,
    evaluate_multiclass_loso,
    multiclass_subject_payload,
    multiclass_xgboost_factory,
)
from fallsense.evaluation.multiclass_reporting import (  # noqa: E402
    plot_multiclass_confusion,
    write_multiclass_report,
)
from fallsense.evaluation.results import (  # noqa: E402
    build_results_payload,
    save_results,
    sha256_path,
)
from fallsense.evaluation.splits import make_loso  # noqa: E402
from fallsense.features import extract_handcrafted_features  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("xgboost", "tcn"), required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("RUN", "QUICK-MODE-ONLY"), required=True)
    return parser


def _tcn_factory(
    payload: dict[str, Any],
    input_channels: int,
) -> tuple[MulticlassConfig, MulticlassModelFactory, str]:
    import torch

    from fallsense.models.tcn import MulticlassTCNClassifier, TCNConfig

    evaluation = MulticlassConfig(**cast(dict[str, Any], payload["evaluation"]))
    model_config = TCNConfig(**cast(dict[str, Any], payload["model"]))
    expected_torch = str(cast(dict[str, Any], payload["runtime"])["torch"])
    if torch.__version__ != expected_torch:
        raise RuntimeError(f"config 要求 torch {expected_torch}，實際為 {torch.__version__}")

    def factory(_labels: np.ndarray[Any, np.dtype[np.int64]], seed: int) -> object:
        return MulticlassTCNClassifier(
            input_channels,
            evaluation.num_classes,
            model_config,
            seed=seed,
        )

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return evaluation, cast(MulticlassModelFactory, factory), device


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_payload = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    archive = np.load(args.windows)
    windows = np.asarray(archive["values"], dtype=np.float64)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    groups = np.asarray(archive["groups"], dtype=np.int64)
    if set(np.unique(labels)) != set(range(11)):
        raise ValueError("11-class artifact 必須包含 zero-based labels 0..10")
    split_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    folds = make_loso(groups)
    if args.model == "xgboost":
        gbm_config = MulticlassGBMConfig(**config_payload)
        evaluation = MulticlassConfig(
            num_classes=gbm_config.num_classes,
            inner_folds=gbm_config.inner_folds,
            random_state=gbm_config.random_state,
        )
        model_factory = multiclass_xgboost_factory(gbm_config)
        features = extract_handcrafted_features(windows)
        values = features.values
        feature_names = features.names
        model_name = "XGBoost-handcrafted-11class"
        runtime_device = "CPU"
    else:
        evaluation, model_factory, runtime_device = _tcn_factory(
            config_payload,
            windows.shape[2],
        )
        values = windows
        feature_names = SENSOR_COLUMNS
        model_name = "TCN-raw-window-channel-mask-11class"
    print(
        json.dumps(
            {
                "model": args.model,
                "device": runtime_device,
                "windows": len(values),
                "subjects": len(folds),
                "models_to_fit": len(folds) * (evaluation.inner_folds + 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results = evaluate_multiclass_loso(
        values,
        labels,
        groups,
        folds,
        config=evaluation,
        model_factory=model_factory,
    )
    feature_hash = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    payload = build_results_payload(
        run_id=args.run_id,
        status=args.status,
        model=model_name,
        task="11-class-activity-recognition",
        root=Path.cwd(),
        config=config_payload,
        config_hash=sha256_path(args.config),
        split_checksum=str(split_payload["checksum"]),
        data_manifest_checksum=sha256_path(args.data_manifest),
        feature_count=len(feature_names),
        feature_hash=feature_hash,
        per_subject=multiclass_subject_payload(results),
        aggregate=aggregate_multiclass_results(results, evaluation.num_classes),
        evaluation_protocol=(
            "outer LOSO; inner GroupKFold OOF multinomial calibration; "
            "per-subject macro-F1 over held-out classes with support"
        ),
    )
    payload["label_mapping"] = {str(index): f"activity_{index + 1}" for index in range(11)}
    payload["imbalance"] = {
        "weight": "n_samples / (n_classes * class_count)",
        "scope": "each inner/final train fold only",
    }
    save_results(payload, args.output_dir / "results.json")
    write_multiclass_report(payload, args.output_dir / "report.md")
    plot_multiclass_confusion(payload, args.output_dir / "confusion_matrix.png")
    aggregate = cast(dict[str, Any], payload["aggregate"])
    print(json.dumps(aggregate["macro_f1"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
