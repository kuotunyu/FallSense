from __future__ import annotations

from pathlib import Path

from fallsense.data.profile import profile_dataset
from fallsense.data.synthetic import build_trial


def test_profile_is_computed_from_files(tmp_path: Path) -> None:
    first = tmp_path / "Subject1Activity1Trial1.csv"
    second = tmp_path / "Subject2Activity6Trial1.csv"
    build_trial(1, 1, 1).to_csv(first, index=False)
    build_trial(2, 6, 1).to_csv(second, index=False)
    profile = profile_dataset([first, second])
    assert profile["trial_files"] == 2
    assert profile["parsed_trials"] == 2
    assert profile["sensor_channels"] == 42
    assert profile["canonical_columns"] == 47
    assert profile["tags"] == [1, 6, 11]
    assert profile["failures"] == {}
