"""在與 primary binary baseline 相同 split 上執行 sensor-set ablation。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.data.schema import IMU_AXES, IMU_COLUMNS  # noqa: E402
from fallsense.evaluation.harness import (  # noqa: E402
    GBMConfig,
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
from fallsense.features import extract_handcrafted_features, select_feature_channels  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("RUN", "QUICK-MODE-ONLY"), required=True)
    return parser


def _channels(name: str) -> tuple[str, ...]:
    if name == "core_waist_motion":
        return tuple(f"Waist_{axis}" for axis in IMU_AXES if axis != "Luminosity")
    if name == "all_wearable_imu":
        return IMU_COLUMNS
    raise ValueError(f"未知 channel_set：{name}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    channel_set = str(payload["channel_set"])
    config = GBMConfig(**cast(dict[str, Any], payload["model"]))
    archive = np.load(args.windows)
    windows = np.asarray(archive["values"], dtype=np.float64)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    groups = np.asarray(archive["groups"], dtype=np.int64)
    features = select_feature_channels(
        extract_handcrafted_features(windows),
        _channels(channel_set),
    )
    split_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    results = evaluate_loso(
        features.values,
        labels,
        groups,
        make_loso(groups),
        config=config,
    )
    feature_hash = hashlib.sha256("\n".join(features.names).encode()).hexdigest()
    result_payload = build_results_payload(
        run_id=args.run_id,
        status=args.status,
        model=f"XGBoost-handcrafted-{channel_set}",
        task="binary-fall-detection-ablation",
        root=Path.cwd(),
        config={"channel_set": channel_set, "model": asdict(config)},
        config_hash=sha256_path(args.config),
        split_checksum=str(split_payload["checksum"]),
        data_manifest_checksum=sha256_path(args.data_manifest),
        feature_count=len(features.names),
        feature_hash=feature_hash,
        per_subject=subject_results_payload(results),
        aggregate=aggregate_subject_results(results),
    )
    save_results(result_payload, args.output_dir / "results.json")
    write_markdown_report(result_payload, args.output_dir / "report.md")
    plot_report_figures(result_payload, args.output_dir)
    print(json.dumps(result_payload["aggregate"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
