"""讀取官方雙層 header 或 synthetic 單層 header 的 per-trial CSV。"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fallsense.data.schema import (
    CSV_COLUMNS,
    ID_COLUMNS,
    SENSOR_COLUMNS,
    TAG_COLUMN,
    TIMESTAMP_COLUMN,
)

TRIAL_FILE_PATTERN = re.compile(r"^Subject(\d+)Activity(\d+)Trial(\d+)\.csv$")


def _is_official_two_row_header(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader)
        second = next(reader)
    return bool(first and first[0] == TIMESTAMP_COLUMN and second and second[0] == "")


def _timestamp_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("float64")
    timestamps = pd.to_datetime(series, errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("TimeStamps 含無法解析的值")
    return (timestamps - timestamps.iloc[0]).dt.total_seconds().astype("float64")


def read_trial_csv(path: Path, *, verify_filename_ids: bool = True) -> pd.DataFrame:
    """正規化成 timestamp + 42 sensors + Subject/Activity/Trial + Tag 共 47 欄。"""

    try:
        if _is_official_two_row_header(path):
            frame = pd.read_csv(path, skiprows=2, header=None, names=CSV_COLUMNS)
        else:
            frame = pd.read_csv(path)
            frame = frame.reindex(columns=CSV_COLUMNS)
    except pd.errors.ParserError as error:
        raise ValueError(f"無法解析 CSV：{path}") from error
    if frame.empty:
        raise ValueError(f"CSV 沒有資料列：{path}")

    frame[TIMESTAMP_COLUMN] = _timestamp_seconds(frame[TIMESTAMP_COLUMN])
    for column in SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (*ID_COLUMNS, TAG_COLUMN):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{path.name} 的 {column} 含缺值或非數字")
        frame[column] = values.astype("int64")

    if (
        frame[TIMESTAMP_COLUMN].duplicated().any()
        or not frame[TIMESTAMP_COLUMN].is_monotonic_increasing
    ):
        raise ValueError(f"{path.name} 的 timestamp 必須嚴格遞增")
    if verify_filename_ids:
        match = TRIAL_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"不符合官方 trial filename：{path.name}")
        expected = tuple(int(value) for value in match.groups())
        actual = tuple(int(frame[column].iloc[0]) for column in ID_COLUMNS)
        if actual != expected:
            raise ValueError(f"filename IDs {expected} 與資料 IDs {actual} 不一致")
        if any(frame[column].nunique() != 1 for column in ID_COLUMNS):
            raise ValueError(f"{path.name} 的 ID 欄在 trial 內不是常數")
    return frame.loc[:, CSV_COLUMNS].copy()


def estimated_sample_rate_hz(frame: pd.DataFrame) -> float:
    """以 timestamp delta 中位數估取樣率，對少量 jitter 較穩健。"""

    delta = np.diff(frame[TIMESTAMP_COLUMN].to_numpy(dtype=np.float64))
    if len(delta) == 0 or np.any(delta <= 0):
        raise ValueError("至少需要兩個嚴格遞增 timestamps")
    return float(1.0 / np.median(delta))
