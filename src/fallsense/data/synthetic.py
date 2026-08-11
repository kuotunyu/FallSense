"""產生不含任何真實受試者資料、可重現的 UP-Fall 形狀 fixture。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from fallsense.data.schema import ID_COLUMNS, SENSOR_COLUMNS, TAG_COLUMN, TIMESTAMP_COLUMN

DEFAULT_SEED = 20260719
DEFAULT_SAMPLE_RATE_HZ = 18.4
DEFAULT_ROWS = 72
SUBJECTS = (1, 2, 3)
ACTIVITIES = (1, 6, 11)
TRIALS = (1, 2)
KNOWN_MISSING_TRIAL = (3, 11, 2)
MISSING_CHANNEL_TRIAL = (2, 6, 2)
MISSING_CHANNEL = "Infrared6"


def trial_filename(subject: int, activity: int, trial: int) -> str:
    """回傳官方 per-trial 形式的檔名。"""

    return f"Subject{subject}Activity{activity}Trial{trial}.csv"


def _stable_trial_seed(seed: int, subject: int, activity: int, trial: int) -> int:
    payload = f"{seed}:{subject}:{activity}:{trial}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _row_labels(activity: int, rows: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    if activity > 5:
        return np.full(rows, activity, dtype=np.int64)
    labels = np.full(rows, 6, dtype=np.int64)
    impact_start = rows // 2 - 3
    labels[impact_start : impact_start + 7] = activity
    labels[impact_start + 7 :] = 11
    return labels


def build_trial(
    subject: int,
    activity: int,
    trial: int,
    *,
    seed: int = DEFAULT_SEED,
    rows: int = DEFAULT_ROWS,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """建立單一 synthetic trial；fall trial 僅在 impact 區段使用 fall row label。"""

    rng = np.random.default_rng(_stable_trial_seed(seed, subject, activity, trial))
    delta = np.clip(
        rng.normal(1.0 / sample_rate_hz, 0.0025, size=rows),
        0.8 / sample_rate_hz,
        1.2 / sample_rate_hz,
    )
    timestamps = np.cumsum(delta) - delta[0]
    x = np.arange(rows, dtype=np.float64)
    data: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {TIMESTAMP_COLUMN: timestamps}

    impact = np.exp(-0.5 * ((x - rows / 2) / 1.8) ** 2)
    for channel_index, channel in enumerate(SENSOR_COLUMNS):
        phase = channel_index * 0.17 + subject * 0.11
        baseline = np.sin(x / (7.0 + channel_index % 5) + phase)
        noise = rng.normal(0.0, 0.025, size=rows)
        signal = baseline + noise
        if activity <= 5 and ("Acc" in channel or "Gyro" in channel):
            signal = signal + impact * (2.5 + channel_index % 4)
        data[channel] = signal

    frame = pd.DataFrame(data)
    frame["Subject"] = subject
    frame["Activity"] = activity
    frame["Trial"] = trial
    frame[TAG_COLUMN] = _row_labels(activity, rows)
    return frame[[TIMESTAMP_COLUMN, *SENSOR_COLUMNS, *ID_COLUMNS, TAG_COLUMN]]


def sha256_file(path: Path) -> str:
    """以串流方式計算檔案 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_fixture(output_dir: Path, *, seed: int = DEFAULT_SEED) -> dict[str, object]:
    """產生固定小型 corpus，並回傳帶 checksum 的 manifest。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for subject in SUBJECTS:
        for activity in ACTIVITIES:
            for trial in TRIALS:
                identity = (subject, activity, trial)
                if identity == KNOWN_MISSING_TRIAL:
                    continue
                frame = build_trial(subject, activity, trial, seed=seed)
                missing_channels: list[str] = []
                if identity == MISSING_CHANNEL_TRIAL:
                    frame = frame.drop(columns=[MISSING_CHANNEL])
                    missing_channels.append(MISSING_CHANNEL)
                filename = trial_filename(subject, activity, trial)
                destination = output_dir / filename
                frame.to_csv(destination, index=False, float_format="%.8f", lineterminator="\n")
                files.append(
                    {
                        "path": filename,
                        "sha256": sha256_file(destination),
                        "rows": len(frame),
                        "missing_channels": missing_channels,
                    }
                )

    manifest: dict[str, object] = {
        "fixture": "synthetic-only; no UP-Fall participant data",
        "seed": seed,
        "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
        "expected_combinations": len(SUBJECTS) * len(ACTIVITIES) * len(TRIALS),
        "known_missing": [trial_filename(*KNOWN_MISSING_TRIAL)],
        "files": files,
    }
    manifest_path = output_dir / "fixture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fixture 輸出目錄")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point。"""

    args = _parser().parse_args(argv)
    manifest = generate_fixture(args.output, seed=args.seed)
    files = cast(list[dict[str, object]], manifest["files"])
    print(json.dumps({"output": str(args.output), "files": len(files)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
