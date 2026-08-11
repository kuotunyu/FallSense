"""可追溯 results.json 建立與反捏造 guard。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_results_payload(
    *,
    run_id: str,
    status: str,
    model: str,
    task: str,
    root: Path,
    config: Mapping[str, object],
    config_hash: str,
    split_checksum: str,
    data_manifest_checksum: str,
    feature_count: int,
    feature_hash: str,
    per_subject: list[dict[str, object]],
    aggregate: Mapping[str, object],
    evaluation_protocol: str = "outer LOSO; inner GroupKFold OOF sigmoid calibration and threshold",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model": model,
        "task": task,
        "git_commit": git_commit(root),
        "config": dict(config),
        "config_hash": config_hash,
        "split_checksum": split_checksum,
        "data_manifest_checksum": data_manifest_checksum,
        "feature_count": feature_count,
        "feature_hash": feature_hash,
        "evaluation_protocol": evaluation_protocol,
        "per_subject": per_subject,
        "aggregate": dict(aggregate),
    }
    validate_results_payload(payload)
    return payload


def _walk_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def validate_results_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "run_id",
        "status",
        "git_commit",
        "config_hash",
        "split_checksum",
        "data_manifest_checksum",
        "per_subject",
        "aggregate",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"results 缺必要欄位：{sorted(missing)}")
    if payload["status"] not in {"RUN", "NOT RUN", "QUICK-MODE-ONLY"}:
        raise ValueError("非法 experiment status")
    if any("accuracy" in key.casefold() for key in _walk_keys(dict(payload))):
        raise ValueError("results 不得包含 accuracy headline")
    per_subject = payload["per_subject"]
    if payload["status"] == "RUN" and (not isinstance(per_subject, list) or not per_subject):
        raise ValueError("RUN results 必須包含 per-subject table")
    aggregate = payload["aggregate"]
    if payload["status"] == "RUN" and (not isinstance(aggregate, dict) or not aggregate):
        raise ValueError("RUN results 不得只有 pooled score")
    if isinstance(aggregate, dict):
        task = str(payload.get("task", ""))
        required_metrics = (
            {"macro_f1"}
            if "11-class" in task
            else {"macro_f1", "fall_recall", "precision", "auprc", "false_positive_rate"}
        )
        if payload["status"] == "RUN" and not required_metrics <= aggregate.keys():
            missing_metrics = sorted(required_metrics - aggregate.keys())
            raise ValueError(f"RUN results 缺少 headline metrics：{missing_metrics}")


def save_results(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
