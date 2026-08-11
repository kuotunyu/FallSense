"""以 ONNX-backed classifier 重跑 binary LOSO，比對已登錄 native metrics。"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.deployment.runtime import create_onnx_session  # noqa: E402
from fallsense.evaluation.harness import (  # noqa: E402
    GBMConfig,
    ProbabilisticClassifier,
    SubjectResult,
    aggregate_subject_results,
    evaluate_loso,
    xgboost_factory,
)
from fallsense.evaluation.splits import make_loso  # noqa: E402
from fallsense.features import extract_handcrafted_features  # noqa: E402
from fallsense.models.tcn import TCNClassifier, TCNConfig  # noqa: E402

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
METRICS = ("macro_f1", "fall_recall", "precision", "auprc", "false_positive_rate")
THRESHOLD_METRICS = ("macro_f1", "fall_recall", "precision", "false_positive_rate")


class _TCNProbabilityModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, values: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(values, channel_mask))


class _OnnxXGBClassifier:
    def __init__(self, config: GBMConfig, labels: IntArray, seed: int, diffs: list[float]) -> None:
        self.native = xgboost_factory(config)(labels, seed)
        self.feature_count = 0
        self.session: Any = None
        self.diffs = diffs

    def fit(self, features: FloatArray, labels: IntArray) -> _OnnxXGBClassifier:
        self.native.fit(features, labels)
        self.feature_count = features.shape[1]
        converted = convert_xgboost(
            self.native,
            initial_types=[("features", FloatTensorType([None, self.feature_count]))],
            target_opset=15,
        )
        self.session = create_onnx_session(converted.SerializeToString())
        return self

    def predict_proba(self, features: FloatArray) -> FloatArray:
        native = np.asarray(self.native.predict_proba(features), dtype=np.float64)
        outputs = self.session.run(None, {"features": features.astype(np.float32)})
        onnx_probability = np.asarray(outputs[1], dtype=np.float64)
        self.diffs.append(float(np.max(np.abs(native - onnx_probability))))
        return onnx_probability


class _OnnxTCNClassifier:
    def __init__(
        self,
        config: TCNConfig,
        input_channels: int,
        seed: int,
        diffs: list[float],
    ) -> None:
        self.native = TCNClassifier(input_channels, config, seed=seed)
        self.input_channels = input_channels
        self.config = config
        self.session: Any = None
        self.diffs = diffs

    def fit(self, features: FloatArray, labels: IntArray) -> _OnnxTCNClassifier:
        self.native.fit(features, labels)
        if self.native.model_ is None:
            raise AssertionError("TCN fit 後無 model")
        model = self.native.model_
        model.cpu().eval()
        probability_model = _TCNProbabilityModule(model).eval()
        buffer = io.BytesIO()
        example_values = torch.zeros((2, features.shape[1], features.shape[2]))
        example_mask = torch.ones((2, features.shape[2]), dtype=torch.bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", torch.jit.TracerWarning)  # type: ignore[attr-defined]
            torch.onnx.export(
                probability_model,
                (example_values, example_mask),
                buffer,  # type: ignore[arg-type]
                input_names=["values", "channel_mask"],
                output_names=["probability"],
                dynamic_axes={
                    "values": {0: "batch"},
                    "channel_mask": {0: "batch"},
                    "probability": {0: "batch"},
                },
                opset_version=15,
                dynamo=False,
            )
        model.to(self.config.device)
        self.session = create_onnx_session(buffer.getvalue())
        return self

    def predict_proba(self, features: FloatArray) -> FloatArray:
        native = self.native.predict_proba(features)
        values, masks = self.native._transform(features)
        positive = np.asarray(
            self.session.run(
                None,
                {"values": values.numpy(), "channel_mask": masks.numpy()},
            )[0],
            dtype=np.float64,
        )
        onnx_probability = np.column_stack((1.0 - positive, positive))
        self.diffs.append(float(np.max(np.abs(native - onnx_probability))))
        return onnx_probability.astype(np.float64)


def _registered_metric_diffs(
    results: tuple[SubjectResult, ...],
    path: Path,
) -> dict[str, float]:
    registered = json.loads(path.read_text(encoding="utf-8"))["per_subject"]
    by_subject = {int(row["subject"]): row for row in registered}
    return {
        metric: max(
            abs(float(getattr(result, metric)) - float(by_subject[result.subject][metric]))
            for result in results
        )
        for metric in METRICS
    }


def _payload(
    name: str,
    results: tuple[SubjectResult, ...],
    diffs: list[float],
    metric_diffs: dict[str, float],
    probability_tolerance: float,
    threshold_metric_tolerance: float,
    auprc_tolerance: float,
) -> dict[str, object]:
    threshold_metric_diff = max(metric_diffs[metric] for metric in THRESHOLD_METRICS)
    return {
        "model": name,
        "max_probability_abs_diff": max(diffs),
        "registered_metric_max_abs_diff": metric_diffs,
        "max_threshold_metric_abs_diff": threshold_metric_diff,
        "max_auprc_abs_diff": metric_diffs["auprc"],
        "probability_tolerance": probability_tolerance,
        "threshold_metric_tolerance": threshold_metric_tolerance,
        "auprc_tolerance": auprc_tolerance,
        "passed": (
            max(diffs) < probability_tolerance
            and threshold_metric_diff < threshold_metric_tolerance
            and metric_diffs["auprc"] < auprc_tolerance
        ),
        "onnx_aggregate": aggregate_subject_results(results),
    }


def _write_report(payload: dict[str, object], path: Path) -> None:
    xgb = cast(dict[str, Any], payload["xgboost"])
    tcn = cast(dict[str, Any], payload["tcn"])
    lines = [
        "# ONNX held-out parity",
        "",
        "ONNX-backed classifiers 重跑與已登錄 runs 相同的 outer LOSO / "
        "inner grouped calibration protocol。",
        "",
        "| model | probability diff | threshold-metric diff | AUPRC diff | result |",
        "|---|---:|---:|---:|---|",
        f"| XGBoost | {xgb['max_probability_abs_diff']:.8g} | "
        f"{xgb['max_threshold_metric_abs_diff']:.8g} | {xgb['max_auprc_abs_diff']:.8g} | "
        f"{'PASS' if xgb['passed'] else 'FAIL'} |",
        f"| TCN | {tcn['max_probability_abs_diff']:.8g} | "
        f"{tcn['max_threshold_metric_abs_diff']:.8g} | {tcn['max_auprc_abs_diff']:.8g} | "
        f"{'PASS' if tcn['passed'] else 'FAIL'} |",
        "",
        "Probability tolerance = 1e-5；threshold-metric tolerance = 1e-4；"
        "rank-sensitive AUPRC tolerance = 0.005。",
        "TCN probability 近似 ties 在 ONNX float32 kernels 可能有極小排序變動，"
        "因此 AUPRC 單獨設定 tolerance，不與 threshold-based metrics 混用。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--xgb-results", type=Path, required=True)
    parser.add_argument("--tcn-results", type=Path, required=True)
    parser.add_argument("--export-config", type=Path, default=Path("configs/export.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/export"))
    args = parser.parse_args()
    archive = np.load(args.windows)
    windows = np.asarray(archive["values"], dtype=np.float64)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    groups = np.asarray(archive["groups"], dtype=np.int64)
    folds = make_loso(groups)
    xgb_config = GBMConfig(
        **cast(
            dict[str, Any],
            yaml.safe_load(Path("configs/gbm_xgboost.yaml").read_text(encoding="utf-8")),
        )
    )
    tcn_yaml = cast(
        dict[str, Any],
        yaml.safe_load(Path("configs/tcn_binary.yaml").read_text(encoding="utf-8")),
    )
    tcn_config = TCNConfig(**cast(dict[str, Any], tcn_yaml["model"]))
    tcn_evaluation = cast(dict[str, Any], tcn_yaml["evaluation"])
    tcn_harness = GBMConfig(
        random_state=int(tcn_evaluation["random_state"]),
        inner_folds=int(tcn_evaluation["inner_folds"]),
        calibration=str(tcn_evaluation["calibration"]),
    )
    export_config = cast(
        dict[str, Any],
        yaml.safe_load(args.export_config.read_text(encoding="utf-8")),
    )
    parity_config = cast(dict[str, Any], export_config["parity"])
    probability_tolerance = float(parity_config["probability_max_abs_diff"])
    threshold_metric_tolerance = float(parity_config["threshold_metric_max_abs_diff"])
    auprc_tolerance = float(parity_config["auprc_max_abs_diff"])

    xgb_diffs: list[float] = []
    print("replaying XGBoost LOSO through ONNX", flush=True)

    def xgb_factory(labels_: IntArray, seed: int) -> ProbabilisticClassifier:
        return _OnnxXGBClassifier(xgb_config, labels_, seed, xgb_diffs)

    xgb_results = evaluate_loso(
        extract_handcrafted_features(windows).values,
        labels,
        groups,
        folds,
        config=xgb_config,
        model_factory=xgb_factory,
    )
    xgb_payload = _payload(
        "XGBoost",
        xgb_results,
        xgb_diffs,
        _registered_metric_diffs(xgb_results, args.xgb_results),
        probability_tolerance,
        threshold_metric_tolerance,
        auprc_tolerance,
    )

    tcn_diffs: list[float] = []
    print("replaying TCN LOSO through ONNX", flush=True)

    def tcn_factory(_labels: IntArray, seed: int) -> ProbabilisticClassifier:
        return _OnnxTCNClassifier(tcn_config, windows.shape[2], seed, tcn_diffs)

    tcn_results = evaluate_loso(
        windows,
        labels,
        groups,
        folds,
        config=tcn_harness,
        model_factory=tcn_factory,
    )
    tcn_payload = _payload(
        "TCN",
        tcn_results,
        tcn_diffs,
        _registered_metric_diffs(tcn_results, args.tcn_results),
        probability_tolerance,
        threshold_metric_tolerance,
        auprc_tolerance,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "opset": 15,
        "xgboost": xgb_payload,
        "tcn": tcn_payload,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_report(payload, args.output_dir / "parity.md")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if not bool(xgb_payload["passed"]) or not bool(tcn_payload["passed"]):
        raise SystemExit("ONNX parity tolerance exceeded; diagnostics were written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
