from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebook_is_valid_and_has_quick_full_resume() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/01_sensor_train_colab.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    source = "\n".join(line for cell in notebook["cells"] for line in cell.get("source", []))
    assert "MODE =" in source and "quick" in source
    assert "full" in source
    assert "results.json" in source
    assert "RESUME" in source
    assert "estimated_peak_ram_gb" in source


def test_results_schema_requires_traceability_and_per_subject() -> None:
    schema = json.loads((ROOT / "schemas/results.schema.json").read_text(encoding="utf-8"))
    required = set(schema["required"])
    assert {"git_commit", "config_hash", "split_checksum", "data_manifest_checksum"} <= required
    assert {"per_subject", "aggregate", "status"} <= required
