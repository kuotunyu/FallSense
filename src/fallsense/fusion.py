"""Late-fusion 介面與 privacy-first 執行守門。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class FusionMode(str, Enum):
    SENSOR_ONLY_PRIVACY = "sensor_only_privacy"
    LATE_FUSION = "late_fusion"


class VisionUnavailableError(RuntimeError):
    """使用者要求 fusion，但目前沒有 vision provider。"""


@dataclass(frozen=True)
class TimedProbability:
    probability: FloatArray
    timestamp_seconds: FloatArray


@dataclass(frozen=True)
class FusionConfig:
    sensor_weight: float = 0.75
    alignment_tolerance_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.sensor_weight <= 1.0:
            raise ValueError("sensor_weight 必須介於 0 與 1")
        if self.alignment_tolerance_seconds < 0.0:
            raise ValueError("alignment_tolerance_seconds 必須 >= 0")


@dataclass(frozen=True)
class FusionResult:
    probability: FloatArray
    mode: FusionMode
    vision_used: bool


VisionProvider = Callable[[], TimedProbability]


def _validate_prediction(prediction: TimedProbability, *, name: str) -> None:
    if prediction.probability.ndim != 1 or prediction.timestamp_seconds.ndim != 1:
        raise ValueError(f"{name} probability/timestamp 必須是一維")
    if prediction.probability.shape != prediction.timestamp_seconds.shape:
        raise ValueError(f"{name} probability 與 timestamp 長度不同")
    if not np.all(np.isfinite(prediction.probability)):
        raise ValueError(f"{name} probability 含非有限值")
    if np.any((prediction.probability < 0.0) | (prediction.probability > 1.0)):
        raise ValueError(f"{name} probability 必須介於 0 與 1")
    if not np.all(np.isfinite(prediction.timestamp_seconds)):
        raise ValueError(f"{name} timestamp 含非有限值")


class LateFusionEngine:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config if config is not None else FusionConfig()

    def predict(
        self,
        sensor: TimedProbability,
        *,
        privacy_mode: bool,
        vision_provider: VisionProvider | None = None,
    ) -> FusionResult:
        _validate_prediction(sensor, name="sensor")
        if privacy_mode:
            # 不 copy、不取用 provider；呼叫端可驗證 object identity 與 bytes。
            return FusionResult(sensor.probability, FusionMode.SENSOR_ONLY_PRIVACY, False)
        if vision_provider is None:
            raise VisionUnavailableError("vision provider 未安裝，無法關閉 privacy mode")

        vision = vision_provider()
        _validate_prediction(vision, name="vision")
        if sensor.probability.shape != vision.probability.shape:
            raise ValueError("sensor/vision window 數量不同，拒絕 fusion")
        offset = np.abs(sensor.timestamp_seconds - vision.timestamp_seconds)
        if np.any(offset > self.config.alignment_tolerance_seconds):
            raise ValueError(
                "sensor/vision timestamp 未對齊："
                f"max offset={float(np.max(offset)):.6f}s > "
                f"{self.config.alignment_tolerance_seconds:.6f}s"
            )
        probability = (
            self.config.sensor_weight * sensor.probability
            + (1.0 - self.config.sensor_weight) * vision.probability
        )
        return FusionResult(probability, FusionMode.LATE_FUSION, True)
