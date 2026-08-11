from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fallsense.models.tcn import TCNClassifier, TCNConfig, TemporalConvNet  # noqa: E402


def test_masked_channel_has_bit_identical_output() -> None:
    torch.manual_seed(7)
    config = TCNConfig(hidden_channels=4, levels=2, dropout=0.0, epochs=1, device="cpu")
    model = TemporalConvNet(3, config).eval()
    values = torch.randn(2, 12, 3)
    mask = torch.tensor([[True, False, True], [True, False, True]])
    perturbed = values.clone()
    perturbed[:, :, 1] = torch.linspace(-1_000_000.0, 1_000_000.0, 12)
    with torch.inference_mode():
        original_output = model(values, mask)
        perturbed_output = model(perturbed, mask)
    assert torch.equal(original_output, perturbed_output)


def test_classifier_fits_scaler_and_weight_from_its_train_fold_only() -> None:
    rng = np.random.default_rng(11)
    train = rng.normal(size=(12, 10, 3)).astype(np.float64)
    train[:, :, 2] = np.nan
    labels = np.asarray([0] * 9 + [1] * 3, dtype=np.int64)
    config = TCNConfig(hidden_channels=4, levels=1, dropout=0.0, epochs=1, device="cpu")
    classifier = TCNClassifier(3, config, seed=3).fit(train, labels)
    assert classifier.positive_weight_ == 3.0
    assert classifier.train_supported_ is not None
    assert classifier.train_supported_.tolist() == [True, True, False]
    expected_mean = np.mean(train[:, :, :2], axis=(0, 1))
    assert classifier.mean_ is not None
    np.testing.assert_allclose(classifier.mean_[:2], expected_mean[:2])
    assert classifier.mean_[2] == 0.0
