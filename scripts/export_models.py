"""訓練 all-subject deployment fits 並匯出 XGBoost/TCN bundles。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import torch
import xgboost as xgb
import yaml
from numpy.typing import NDArray
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from safetensors.torch import save_file
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.evaluation.harness import (  # noqa: E402
    GBMConfig,
    choose_threshold,
    fit_sigmoid_calibrator,
    threshold_curve,
)
from fallsense.evaluation.results import git_commit, sha256_path  # noqa: E402
from fallsense.features import extract_handcrafted_features  # noqa: E402
from fallsense.models.tcn import TCNClassifier, TCNConfig  # noqa: E402

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class _TCNProbabilityModule(torch.nn.Module):
    """Sigmoid 併入 ONNX graph，避免 runtime 以不同 precision 重算。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, values: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(values, channel_mask))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--xgb-config", type=Path, default=Path("configs/gbm_xgboost.yaml"))
    parser.add_argument("--tcn-config", type=Path, default=Path("configs/tcn_binary.yaml"))
    parser.add_argument("--export-config", type=Path, default=Path("configs/export.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    return parser


def _xgb_model(config: GBMConfig, labels: IntArray, seed: int) -> xgb.XGBClassifier:
    negatives = int(np.sum(labels == 0))
    positives = int(np.sum(labels == 1))
    return xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        n_jobs=config.n_jobs,
        random_state=seed,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=negatives / positives,
    )


def _deployment_calibration(
    values: FloatArray,
    labels: IntArray,
    groups: IntArray,
    *,
    folds: int,
    seed: int,
    factory: Any,
) -> tuple[float, float, float]:
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for number, (train, valid) in enumerate(splitter.split(values, labels, groups=groups)):
        model = factory(labels[train], seed + number)
        model.fit(values[train], labels[train])
        oof[valid] = np.asarray(model.predict_proba(values[valid]), dtype=np.float64)[:, 1]
    calibrator = fit_sigmoid_calibrator(oof, labels)
    threshold = choose_threshold(threshold_curve(labels, calibrator.predict(oof))).threshold
    return calibrator.coefficient, calibrator.intercept, threshold


def _save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _export_xgb(
    features: FloatArray,
    feature_names: tuple[str, ...],
    labels: IntArray,
    groups: IntArray,
    config: GBMConfig,
    export_payload: dict[str, Any],
    output: Path,
) -> dict[str, object]:
    coefficient, intercept, threshold = _deployment_calibration(
        features,
        labels,
        groups,
        folds=int(export_payload["calibration_folds"]),
        seed=int(export_payload["random_state"]),
        factory=lambda fold_labels, seed: _xgb_model(config, fold_labels, seed),
    )
    model = _xgb_model(config, labels, int(export_payload["random_state"]))
    model.fit(features, labels)
    output.mkdir(parents=True, exist_ok=True)
    native_path = output / "model.json"
    onnx_path = output / "model.onnx"
    model.save_model(native_path)
    converted = convert_xgboost(
        model,
        initial_types=[("features", FloatTensorType([None, features.shape[1]]))],
        target_opset=int(export_payload["opset"]),
    )
    onnx.checker.check_model(converted)
    onnx.save_model(converted, onnx_path)
    metadata: dict[str, object] = {
        "model": "XGBoost-handcrafted-binary",
        "deployment_fit_scope": "all 17 subjects; not an evaluation score",
        "input": {"shape": [None, features.shape[1]], "dtype": "float32"},
        "feature_names": list(feature_names),
        "feature_hash": hashlib.sha256("\n".join(feature_names).encode()).hexdigest(),
        "calibration": {
            "method": "sigmoid on all-subject GroupKFold OOF probabilities",
            "coefficient": coefficient,
            "intercept": intercept,
            "threshold": threshold,
        },
        "opset": int(export_payload["opset"]),
        "xgboost": xgb.__version__,
    }
    _save_json(metadata, output / "metadata.json")
    return metadata


def _export_tcn(
    windows: FloatArray,
    labels: IntArray,
    groups: IntArray,
    config: TCNConfig,
    export_payload: dict[str, Any],
    output: Path,
) -> dict[str, object]:
    coefficient, intercept, threshold = _deployment_calibration(
        windows,
        labels,
        groups,
        folds=int(export_payload["calibration_folds"]),
        seed=int(export_payload["random_state"]),
        factory=lambda _fold_labels, seed: TCNClassifier(windows.shape[2], config, seed=seed),
    )
    classifier = TCNClassifier(
        windows.shape[2],
        config,
        seed=int(export_payload["random_state"]),
    ).fit(windows, labels)
    if classifier.model_ is None or classifier.mean_ is None or classifier.scale_ is None:
        raise AssertionError("TCN deployment fit 不完整")
    if classifier.train_supported_ is None:
        raise AssertionError("TCN train-supported mask 不完整")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "model.safetensors"
    onnx_path = output / "model.onnx"
    # Export 統一在 CPU 完成，避免 trace inputs 與訓練後 CUDA weights 不同 device。
    classifier.model_.cpu()
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in classifier.model_.state_dict().items()
    }
    save_file(state, state_path)
    example_values = torch.zeros((2, windows.shape[1], windows.shape[2]), dtype=torch.float32)
    example_mask = torch.ones((2, windows.shape[2]), dtype=torch.bool)
    classifier.model_.eval()
    probability_model = _TCNProbabilityModule(classifier.model_).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", torch.jit.TracerWarning)  # type: ignore[attr-defined]
        torch.onnx.export(
            probability_model,
            (example_values, example_mask),
            onnx_path,
            input_names=["values", "channel_mask"],
            output_names=["probability"],
            dynamic_axes={
                "values": {0: "batch"},
                "channel_mask": {0: "batch"},
                "probability": {0: "batch"},
            },
            opset_version=int(export_payload["opset"]),
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(onnx_path))
    metadata = {
        "model": "channel-masked-TCN-binary-secondary",
        "deployment_fit_scope": "all 17 subjects; not an evaluation score",
        "selection_note": "secondary parity artifact; XGBoost is the primary deployment candidate",
        "input": {"shape": [None, windows.shape[1], windows.shape[2]], "dtype": "float32"},
        "config": asdict(config),
        "normalization": {
            "mean": classifier.mean_.tolist(),
            "scale": classifier.scale_.tolist(),
            "train_supported": classifier.train_supported_.tolist(),
        },
        "calibration": {
            "method": "sigmoid on all-subject GroupKFold OOF probabilities",
            "coefficient": coefficient,
            "intercept": intercept,
            "threshold": threshold,
        },
        "opset": int(export_payload["opset"]),
        "torch": torch.__version__,
    }
    _save_json(metadata, output / "metadata.json")
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archive = np.load(args.windows)
    windows = np.asarray(archive["values"], dtype=np.float64)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    groups = np.asarray(archive["groups"], dtype=np.int64)
    feature_matrix = extract_handcrafted_features(windows)
    xgb_payload = cast(dict[str, Any], yaml.safe_load(args.xgb_config.read_text(encoding="utf-8")))
    tcn_payload = cast(dict[str, Any], yaml.safe_load(args.tcn_config.read_text(encoding="utf-8")))
    export_payload = cast(
        dict[str, Any],
        yaml.safe_load(args.export_config.read_text(encoding="utf-8")),
    )
    _export_xgb(
        feature_matrix.values,
        feature_matrix.names,
        labels,
        groups,
        GBMConfig(**xgb_payload),
        export_payload,
        args.output_dir / "xgb_binary",
    )
    _export_tcn(
        windows,
        labels,
        groups,
        TCNConfig(**cast(dict[str, Any], tcn_payload["model"])),
        export_payload,
        args.output_dir / "tcn_binary",
    )
    manifest_path = args.output_dir / "manifest.json"
    files = sorted(
        path for path in args.output_dir.rglob("*") if path.is_file() and path != manifest_path
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "git_commit": git_commit(Path.cwd()),
        "opset": int(export_payload["opset"]),
        "source_split_checksum": json.loads(args.split_manifest.read_text(encoding="utf-8"))[
            "checksum"
        ],
        "data_manifest_checksum": sha256_path(args.data_manifest),
        "configs": {
            "xgb": sha256_path(args.xgb_config),
            "tcn": sha256_path(args.tcn_config),
            "export": sha256_path(args.export_config),
        },
        "files": {
            path.relative_to(args.output_dir).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
            for path in files
        },
    }
    _save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
