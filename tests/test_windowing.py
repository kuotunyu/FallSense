from __future__ import annotations

from pathlib import Path

import numpy as np

from fallsense.data.io import read_trial_csv
from fallsense.data.schema import SENSOR_COLUMNS
from fallsense.data.synthetic import build_trial
from fallsense.data.visualize import plot_trial_alignment
from fallsense.data.windowing import WindowConfig, window_trial


def test_window_count_matches_hand_calculation() -> None:
    frame = build_trial(1, 1, 1, rows=72)
    config = WindowConfig(window_size_samples=20, stride_samples=10)
    batch = window_trial(frame, config, source_path="trial.csv")
    assert len(batch.labels) == 6
    assert batch.values.shape == (6, 20, 42)


def test_trial_shorter_than_window_returns_empty_batch() -> None:
    frame = build_trial(1, 6, 1, rows=10)
    config = WindowConfig(window_size_samples=20, stride_samples=10)
    batch = window_trial(frame, config, source_path="short.csv")
    assert batch.values.shape == (0, 20, 42)
    assert batch.metadata == ()


def test_missing_channel_is_preserved_and_masked() -> None:
    frame = build_trial(1, 6, 1, rows=30)
    missing_index = SENSOR_COLUMNS.index("Infrared6")
    frame["Infrared6"] = np.nan
    config = WindowConfig(window_size_samples=20, stride_samples=10)
    batch = window_trial(frame, config, source_path="missing.csv")
    assert np.isnan(batch.values[:, :, missing_index]).all()
    assert not batch.channel_mask[:, missing_index].any()


def test_tag20_drop_policy_removes_affected_windows() -> None:
    frame = build_trial(1, 6, 1, rows=30)
    frame["Tag"] = 20
    config = WindowConfig(
        window_size_samples=20,
        stride_samples=10,
        tag20_policy="drop_window",
    )
    batch = window_trial(frame, config, source_path="tag20.csv")
    assert len(batch.labels) == 0


def test_multiclass_center_label_is_zero_based_activity_tag() -> None:
    frame = build_trial(1, 11, 1, rows=30)
    frame["Tag"] = 11
    config = WindowConfig(
        window_size_samples=20,
        stride_samples=10,
        label_strategy="center",
        task="multiclass_11",
    )
    batch = window_trial(frame, config, source_path="activity11.csv")
    assert batch.labels.tolist() == [10, 10]


def test_binary_config_checksum_remains_backward_compatible() -> None:
    config = WindowConfig.from_yaml(Path("configs/windowing_50_25.yaml"))
    assert config.checksum == "375ee9328d50ecb3fe0e905020d304492b27a5755898df772fad05f13bf5c51d"


def test_center_label_detects_impact_while_majority_does_not() -> None:
    frame = build_trial(1, 1, 1, rows=72)
    center = window_trial(
        frame,
        WindowConfig(window_size_samples=20, stride_samples=10, label_strategy="center"),
        source_path="fall.csv",
    )
    majority = window_trial(
        frame,
        WindowConfig(window_size_samples=20, stride_samples=10, label_strategy="majority"),
        source_path="fall.csv",
    )
    assert center.labels.sum() > majority.labels.sum()


def test_alignment_plot_is_created(tmp_path) -> None:
    source = tmp_path / "Subject1Activity1Trial1.csv"
    build_trial(1, 1, 1).to_csv(source, index=False)
    output = tmp_path / "alignment.png"
    plot_trial_alignment(read_trial_csv(source), output, title="synthetic fall")
    assert output.stat().st_size > 1_000
