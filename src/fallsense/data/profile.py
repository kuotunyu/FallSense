"""以實際 per-trial CSV 回填資料集常數，不依賴文件硬編碼。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fallsense.data.io import estimated_sample_rate_hz, read_trial_csv
from fallsense.data.schema import SENSOR_COLUMNS, TAG_COLUMN


def profile_dataset(paths: list[Path]) -> dict[str, object]:
    rates: list[float] = []
    rows: list[int] = []
    tags: set[int] = set()
    missing_channel_trials: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    total_bytes = 0
    tag20_trials = 0
    tag20_trial_names: list[str] = []
    for path in sorted(paths):
        total_bytes += path.stat().st_size
        try:
            frame = read_trial_csv(path)
        except (OSError, ValueError) as error:
            failures[path.name] = str(error)
            continue
        rates.append(estimated_sample_rate_hz(frame))
        rows.append(len(frame))
        tags.update(int(value) for value in frame[TAG_COLUMN].unique())
        if (frame[TAG_COLUMN] == 20).any():
            tag20_trials += 1
            tag20_trial_names.append(path.name)
        missing = [column for column in SENSOR_COLUMNS if frame[column].isna().all()]
        if missing:
            missing_channel_trials[path.name] = missing
    if not rates:
        raise ValueError("沒有可成功解析的 trial")
    return {
        "trial_files": len(paths),
        "parsed_trials": len(rates),
        "total_bytes": total_bytes,
        "sensor_channels": len(SENSOR_COLUMNS),
        "canonical_columns": 47,
        "rows": {
            "min": min(rows),
            "median": float(np.median(rows)),
            "max": max(rows),
            "total": sum(rows),
        },
        "sample_rate_hz": {
            "min": min(rates),
            "median": float(np.median(rates)),
            "max": max(rates),
        },
        "tags": sorted(tags),
        "tag20_trials": tag20_trials,
        "tag20_trial_names": tag20_trial_names,
        "missing_channel_trials": missing_channel_trials,
        "missing_channel_counts": {
            channel: sum(channel in missing for missing in missing_channel_trials.values())
            for channel in SENSOR_COLUMNS
            if any(channel in missing for missing in missing_channel_trials.values())
        },
        "failures": failures,
    }


def save_profile(profile: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
