# ruff: noqa: E402, I001

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")


ARTIFACTS = Path("artifacts/models")


@pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="export 尚未產生")
def test_exported_models_run_without_importing_torch() -> None:
    code = """
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from fallsense.data.io import read_trial_csv
from fallsense.data.windowing import WindowConfig, window_trial
from fallsense.deployment.runtime import create_onnx_session, predict_tcn_probability
frame = read_trial_csv(Path('tests/fixtures/upfall_synthetic/Subject1Activity1Trial1.csv'))
windows = window_trial(
    frame,
    WindowConfig(window_size_samples=50, stride_samples=25),
    source_path='synthetic-fall.csv',
).values[:3]
metadata = json.loads(Path('artifacts/models/tcn_binary/metadata.json').read_text(encoding='utf-8'))
session = create_onnx_session(Path('artifacts/models/tcn_binary/model.onnx'))
assert predict_tcn_probability(session, windows, metadata).shape == (3,)
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
