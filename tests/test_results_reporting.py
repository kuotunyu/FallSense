from __future__ import annotations

from pathlib import Path

import pytest

from fallsense.evaluation.reporting import write_markdown_report
from fallsense.evaluation.results import validate_results_payload


def _payload() -> dict[str, object]:
    aggregate = {
        key: {"mean": 0.5, "std": 0.1}
        for key in ("macro_f1", "fall_recall", "precision", "auprc", "false_positive_rate")
    }
    row = {
        "subject": 1,
        "threshold": 0.4,
        "macro_f1": 0.5,
        "fall_recall": 0.6,
        "precision": 0.4,
        "auprc": 0.55,
        "false_positive_rate": 0.1,
    }
    return {
        "run_id": "test",
        "status": "RUN",
        "git_commit": "abc",
        "config_hash": "cfg",
        "split_checksum": "split",
        "data_manifest_checksum": "data",
        "per_subject": [row],
        "aggregate": aggregate,
    }


def test_results_guard_rejects_accuracy_and_pooled_only() -> None:
    payload = _payload()
    payload["accuracy"] = 0.99
    with pytest.raises(ValueError, match="accuracy"):
        validate_results_payload(payload)
    pooled = _payload()
    pooled["per_subject"] = []
    with pytest.raises(ValueError, match="per-subject"):
        validate_results_payload(pooled)


def test_markdown_report_contains_required_headlines(tmp_path: Path) -> None:
    payload = _payload()
    validate_results_payload(payload)
    output = tmp_path / "report.md"
    write_markdown_report(payload, output)
    text = output.read_text(encoding="utf-8")
    assert "macro-F1" in text
    assert "fall recall" in text
    assert "AUPRC" in text
    assert "Per-subject" in text
    assert "accuracy" not in text.casefold()
