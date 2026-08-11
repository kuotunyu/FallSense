"""執行 binary handcrafted-feature XGBoost nested LOSO baseline。"""

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
from fallsense.features import extract_handcrafted_features  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/gbm_xgboost.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("RUN", "QUICK-MODE-ONLY"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_payload = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    config = GBMConfig(**config_payload)
    archive = np.load(args.windows)
    windows = np.asarray(archive["values"], dtype=np.float64)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    groups = np.asarray(archive["groups"], dtype=np.int64)
    feature_matrix = extract_handcrafted_features(windows)
    feature_hash = hashlib.sha256("\n".join(feature_matrix.names).encode()).hexdigest()
    split_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    folds = make_loso(groups)
    results = evaluate_loso(feature_matrix.values, labels, groups, folds, config=config)
    payload = build_results_payload(
        run_id=args.run_id,
        status=args.status,
        model="XGBoost-handcrafted",
        task="binary-fall-detection",
        root=Path.cwd(),
        config=asdict(config),
        config_hash=sha256_path(args.config),
        split_checksum=str(split_payload["checksum"]),
        data_manifest_checksum=sha256_path(args.data_manifest),
        feature_count=len(feature_matrix.names),
        feature_hash=feature_hash,
        per_subject=subject_results_payload(results),
        aggregate=aggregate_subject_results(results),
    )
    save_results(payload, args.output_dir / "results.json")
    write_markdown_report(payload, args.output_dir / "report.md")
    plot_report_figures(payload, args.output_dir)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
