from __future__ import annotations

import numpy as np
import pytest

from fallsense.fusion import (
    FusionConfig,
    FusionMode,
    LateFusionEngine,
    TimedProbability,
    VisionUnavailableError,
)


def _sensor() -> TimedProbability:
    return TimedProbability(
        probability=np.asarray([0.1, 0.8, 0.3], dtype=np.float64),
        timestamp_seconds=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
    )


def test_privacy_mode_is_bit_identical_and_never_calls_vision() -> None:
    sensor = _sensor()
    original_bytes = sensor.probability.tobytes()

    def forbidden_provider() -> TimedProbability:
        raise AssertionError("privacy mode 不得呼叫 vision provider")

    result = LateFusionEngine().predict(
        sensor,
        privacy_mode=True,
        vision_provider=forbidden_provider,
    )
    assert result.mode is FusionMode.SENSOR_ONLY_PRIVACY
    assert not result.vision_used
    assert result.probability is sensor.probability
    assert result.probability.tobytes() == original_bytes


def test_fusion_requires_explicit_vision_provider() -> None:
    with pytest.raises(VisionUnavailableError, match="vision provider"):
        LateFusionEngine().predict(_sensor(), privacy_mode=False)


def test_late_fusion_rejects_timestamp_shift() -> None:
    vision = TimedProbability(
        probability=np.asarray([0.2, 0.4, 0.6], dtype=np.float64),
        timestamp_seconds=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
    )
    engine = LateFusionEngine(FusionConfig(alignment_tolerance_seconds=0.25))
    with pytest.raises(ValueError, match="timestamp 未對齊"):
        engine.predict(_sensor(), privacy_mode=False, vision_provider=lambda: vision)


def test_late_fusion_uses_fixed_weight_after_alignment() -> None:
    sensor = _sensor()
    vision = TimedProbability(
        probability=np.asarray([0.5, 0.4, 0.1], dtype=np.float64),
        timestamp_seconds=sensor.timestamp_seconds.copy(),
    )
    result = LateFusionEngine(FusionConfig(sensor_weight=0.75)).predict(
        sensor,
        privacy_mode=False,
        vision_provider=lambda: vision,
    )
    np.testing.assert_allclose(result.probability, [0.2, 0.7, 0.25], rtol=0.0, atol=1e-15)
    assert result.mode is FusionMode.LATE_FUSION
    assert result.vision_used
