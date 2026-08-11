# ruff: noqa: E402, I001

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
safetensors = pytest.importorskip("safetensors.torch")
xgb = pytest.importorskip("xgboost")

from fallsense.deployment.runtime import (
    create_onnx_session,
    predict_tcn_probability,
    predict_xgb_probability,
    prepare_tcn_inputs,
)
from fallsense.features import extract_handcrafted_features
from fallsense.models.tcn import TCNConfig, TemporalConvNet


ARTIFACTS = Path("artifacts/models")
TOLERANCE = 1e-5


@pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="export 尚未產生")
@pytest.mark.parametrize(
    "archive_path",
    (
        Path("data/processed/synthetic_50_25/windows.npz"),
        Path("data/processed/real_50_25_center/windows.npz"),
    ),
)
def test_native_and_onnx_probability_parity(archive_path: Path) -> None:
    windows = np.asarray(np.load(archive_path)["values"][:64], dtype=np.float64)

    native_xgb = xgb.XGBClassifier()
    native_xgb.load_model(ARTIFACTS / "xgb_binary/model.json")
    features = extract_handcrafted_features(windows).values
    native_xgb_probability = np.asarray(native_xgb.predict_proba(features))[:, 1]
    xgb_session = create_onnx_session(ARTIFACTS / "xgb_binary/model.onnx")
    onnx_xgb_probability = predict_xgb_probability(xgb_session, features)
    assert float(np.max(np.abs(native_xgb_probability - onnx_xgb_probability))) < TOLERANCE

    metadata = json.loads((ARTIFACTS / "tcn_binary/metadata.json").read_text(encoding="utf-8"))
    model = TemporalConvNet(42, TCNConfig(**metadata["config"])).eval()
    model.load_state_dict(safetensors.load_file(ARTIFACTS / "tcn_binary/model.safetensors"))
    values, masks = prepare_tcn_inputs(windows, metadata)
    with torch.inference_mode():
        native_tcn_probability = torch.sigmoid(
            model(torch.from_numpy(values), torch.from_numpy(masks))
        ).numpy()
    tcn_session = create_onnx_session(ARTIFACTS / "tcn_binary/model.onnx")
    onnx_tcn_probability = predict_tcn_probability(tcn_session, windows, metadata)
    assert float(np.max(np.abs(native_tcn_probability - onnx_tcn_probability))) < TOLERANCE
