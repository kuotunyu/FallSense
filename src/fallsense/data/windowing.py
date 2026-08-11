"""Config-driven windowing 與 binary label mapping。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from fallsense.data.io import read_trial_csv
from fallsense.data.schema import ID_COLUMNS, SENSOR_COLUMNS, TAG_COLUMN, TIMESTAMP_COLUMN

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
LabelStrategy = Literal["center", "majority", "any_fall", "fall_fraction"]
Tag20Policy = Literal["drop_window", "map_nonfall", "error"]
MissingChannelPolicy = Literal["preserve_nan", "fill_zero", "error"]
WindowTask = Literal["binary", "multiclass_11"]


@dataclass(frozen=True)
class WindowConfig:
    window_size_samples: int
    stride_samples: int
    label_strategy: LabelStrategy = "center"
    fall_fraction_threshold: float = 0.2
    tag20_policy: Tag20Policy = "drop_window"
    missing_channel_policy: MissingChannelPolicy = "preserve_nan"
    task: WindowTask = "binary"

    def __post_init__(self) -> None:
        if self.window_size_samples < 1 or self.stride_samples < 1:
            raise ValueError("window_size_samples 與 stride_samples 必須為正數")
        if not 0.0 <= self.fall_fraction_threshold <= 1.0:
            raise ValueError("fall_fraction_threshold 必須介於 0 與 1")
        if self.task == "multiclass_11" and self.label_strategy not in {"center", "majority"}:
            raise ValueError("11-class 只支援 center 或 majority label strategy")

    @classmethod
    def from_yaml(cls, path: Path) -> WindowConfig:
        payload = cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))
        return cls(**payload)

    @property
    def checksum(self) -> str:
        payload = asdict(self)
        # 保持 M2 已發佈 binary config/split checksum 的向後相容性。
        if self.task == "binary":
            payload.pop("task")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WindowMetadata:
    window_id: str
    source_path: str
    start_row: int
    end_row: int
    start_seconds: float
    end_seconds: float
    subject: int
    activity: int
    trial: int
    raw_tags: tuple[int, ...]


@dataclass(frozen=True)
class WindowBatch:
    values: FloatArray
    labels: IntArray
    groups: IntArray
    channel_mask: BoolArray
    metadata: tuple[WindowMetadata, ...]


def binary_tag(tag: int, policy: Tag20Policy) -> int | None:
    if 1 <= tag <= 5:
        return 1
    if 6 <= tag <= 11:
        return 0
    if tag == 20:
        if policy == "drop_window":
            return None
        if policy == "map_nonfall":
            return 0
    raise ValueError(f"未定義的 tag：{tag}")


def _window_label(tags: IntArray, config: WindowConfig) -> int | None:
    if config.tag20_policy == "drop_window" and np.any(tags == 20):
        return None
    if config.tag20_policy == "error" and np.any(tags == 20):
        raise ValueError("window 含 tag 20，但 config 設為 error")
    if config.task == "multiclass_11":
        if np.any((tags < 1) | ((tags > 11) & (tags != 20))):
            raise ValueError("11-class window 含有效範圍 1..11 以外的 tag")
        if config.tag20_policy == "map_nonfall" and np.any(tags == 20):
            raise ValueError("11-class 不允許將 tag20 武斷映射為單一 ADL class")
        zero_based = tags - 1
        if config.label_strategy == "center":
            return int(zero_based[(len(zero_based) - 1) // 2])
        return int(np.argmax(np.bincount(zero_based, minlength=11)))
    mapped = np.array([binary_tag(int(tag), config.tag20_policy) for tag in tags])
    if np.any(mapped == None):  # noqa: E711
        return None
    binary = mapped.astype(np.int64)
    if config.label_strategy == "center":
        return int(binary[(len(binary) - 1) // 2])
    if config.label_strategy == "majority":
        counts = np.bincount(binary, minlength=2)
        return int(np.argmax(counts))
    if config.label_strategy == "any_fall":
        return int(np.any(binary == 1))
    return int(float(np.mean(binary)) >= config.fall_fraction_threshold)


def window_trial(frame: pd.DataFrame, config: WindowConfig, *, source_path: str) -> WindowBatch:
    """單一 trial 切窗；短 trial 回傳 shape 正確的空 batch。"""

    size = config.window_size_samples
    starts = range(0, max(0, len(frame) - size + 1), config.stride_samples)
    values: list[FloatArray] = []
    labels: list[int] = []
    groups: list[int] = []
    masks: list[BoolArray] = []
    metadata: list[WindowMetadata] = []
    subject, activity, trial = (int(frame[column].iloc[0]) for column in ID_COLUMNS)
    for start in starts:
        end = start + size
        block = frame.iloc[start:end]
        sensors = block.loc[:, SENSOR_COLUMNS].to_numpy(dtype=np.float64)
        if config.missing_channel_policy == "error" and np.isnan(sensors).any():
            raise ValueError(f"{source_path} window {start}:{end} 含 NaN")
        if config.missing_channel_policy == "fill_zero":
            np.nan_to_num(sensors, copy=False, nan=0.0)
        label = _window_label(block[TAG_COLUMN].to_numpy(dtype=np.int64), config)
        if label is None:
            continue
        channel_mask = ~np.all(np.isnan(sensors), axis=0)
        window_id = f"{source_path}#rows={start}:{end}"
        values.append(sensors)
        labels.append(label)
        groups.append(subject)
        masks.append(channel_mask)
        metadata.append(
            WindowMetadata(
                window_id=window_id,
                source_path=source_path,
                start_row=start,
                end_row=end,
                start_seconds=float(block[TIMESTAMP_COLUMN].iloc[0]),
                end_seconds=float(block[TIMESTAMP_COLUMN].iloc[-1]),
                subject=subject,
                activity=activity,
                trial=trial,
                raw_tags=tuple(sorted(int(tag) for tag in block[TAG_COLUMN].unique())),
            )
        )
    if not values:
        return WindowBatch(
            np.empty((0, size, len(SENSOR_COLUMNS)), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, len(SENSOR_COLUMNS)), dtype=np.bool_),
            (),
        )
    return WindowBatch(
        np.stack(values),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
        np.stack(masks),
        tuple(metadata),
    )


def window_files(
    paths: Sequence[Path], config: WindowConfig, *, root: Path | None = None
) -> WindowBatch:
    batches = [
        window_trial(
            read_trial_csv(path),
            config,
            source_path=(path.relative_to(root).as_posix() if root is not None else path.name),
        )
        for path in sorted(paths)
    ]
    nonempty = [batch for batch in batches if len(batch.labels)]
    if not nonempty:
        return WindowBatch(
            np.empty((0, config.window_size_samples, len(SENSOR_COLUMNS)), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, len(SENSOR_COLUMNS)), dtype=np.bool_),
            (),
        )
    return WindowBatch(
        np.concatenate([batch.values for batch in nonempty]),
        np.concatenate([batch.labels for batch in nonempty]),
        np.concatenate([batch.groups for batch in nonempty]),
        np.concatenate([batch.channel_mask for batch in nonempty]),
        tuple(item for batch in nonempty for item in batch.metadata),
    )
