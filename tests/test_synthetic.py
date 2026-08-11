from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from fallsense.data.schema import CSV_COLUMNS
from fallsense.data.synthetic import (
    MISSING_CHANNEL,
    MISSING_CHANNEL_TRIAL,
    generate_fixture,
    sha256_file,
    trial_filename,
)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_fixture_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_fixture(first)
    generate_fixture(second)
    assert _tree_digest(first) == _tree_digest(second)


def test_fixture_contains_declared_missing_cases(tmp_path: Path) -> None:
    manifest = generate_fixture(tmp_path)
    assert manifest["expected_combinations"] == 18
    assert len(manifest["files"]) == 17
    missing_path = trial_filename(*MISSING_CHANNEL_TRIAL)
    frame = pd.read_csv(tmp_path / missing_path)
    assert MISSING_CHANNEL not in frame.columns
    normal = pd.read_csv(tmp_path / "Subject1Activity6Trial1.csv")
    assert tuple(normal.columns) == CSV_COLUMNS


def test_fall_trial_has_short_impact_label_segment(tmp_path: Path) -> None:
    generate_fixture(tmp_path)
    frame = pd.read_csv(tmp_path / "Subject1Activity1Trial1.csv")
    assert set(frame["Activity"]) == {1}
    assert set(frame["Tag"]) == {1, 6, 11}
    assert int((frame["Tag"] == 1).sum()) == 7


def test_committed_fixture_manifest_checksums_match() -> None:
    fixture = Path(__file__).parent / "fixtures" / "upfall_synthetic"
    manifest = json.loads((fixture / "fixture_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        assert sha256_file(fixture / record["path"]) == record["sha256"]
