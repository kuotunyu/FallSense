"""Torch-free ONNX Runtime inference helpers。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def create_onnx_session(
    model: Path | bytes,
    *,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> ort.InferenceSession:
    source: str | bytes = str(model) if isinstance(model, Path) else model
    return ort.InferenceSession(source, providers=list(providers))


def predict_xgb_probability(
    session: ort.InferenceSession,
    features: FloatArray,
) -> FloatArray:
    outputs = session.run(None, {session.get_inputs()[0].name: features.astype(np.float32)})
    probabilities = np.asarray(outputs[1], dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("XGBoost ONNX probabilities 必須是 [N,2]")
    return probabilities[:, 1]


def prepare_tcn_inputs(
    windows: FloatArray,
    metadata: dict[str, Any],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float64)
    scale = np.asarray(metadata["normalization"]["scale"], dtype=np.float64)
    supported = np.asarray(metadata["normalization"]["train_supported"], dtype=np.bool_)
    if windows.ndim != 3 or windows.shape[2] != len(mean):
        raise ValueError("TCN windows shape 與 metadata 不相容")
    sample_mask = np.any(np.isfinite(windows), axis=1)
    mask = sample_mask & supported.reshape(1, -1)
    normalized = (windows - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32), mask


def predict_tcn_probability(
    session: ort.InferenceSession,
    windows: FloatArray,
    metadata: dict[str, Any],
) -> FloatArray:
    values, mask = prepare_tcn_inputs(windows, metadata)
    inputs = {item.name: item for item in session.get_inputs()}
    probabilities = np.asarray(
        session.run(
            None,
            {
                inputs["values"].name: values,
                inputs["channel_mask"].name: mask,
            },
        )[0],
        dtype=np.float64,
    )
    return probabilities
