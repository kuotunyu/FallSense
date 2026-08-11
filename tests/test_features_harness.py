from __future__ import annotations

import numpy as np

from fallsense.evaluation.harness import (
    GBMConfig,
    aggregate_subject_results,
    evaluate_loso,
)
from fallsense.evaluation.splits import make_loso
from fallsense.features import STAT_NAMES, extract_handcrafted_features, select_feature_channels


def test_handcrafted_feature_shape_and_missing_channel() -> None:
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(5, 20, 42))
    windows[:, :, 3] = np.nan
    features = extract_handcrafted_features(windows)
    assert features.values.shape == (5, len(STAT_NAMES) * 42)
    assert len(features.names) == features.values.shape[1]
    missing_positions = [
        index for index, name in enumerate(features.names) if name.endswith("LeftAnkle_GyroX")
    ]
    assert np.isnan(features.values[:, missing_positions]).all()


def test_select_feature_channels_keeps_every_stat_for_requested_channels() -> None:
    windows = np.zeros((2, 20, 42), dtype=np.float64)
    features = extract_handcrafted_features(windows)
    selected = select_feature_channels(features, ("Waist_AccX", "Waist_GyroZ"))
    assert selected.values.shape == (2, len(STAT_NAMES) * 2)
    assert all(name.endswith(("Waist_AccX", "Waist_GyroZ")) for name in selected.names)


class _SimpleClassifier:
    def fit(self, features: np.ndarray, labels: np.ndarray) -> object:
        negative = float(np.mean(features[labels == 0, 0]))
        positive = float(np.mean(features[labels == 1, 0]))
        self.midpoint = (negative + positive) / 2
        self.direction = 1.0 if positive >= negative else -1.0
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        score = 1.0 / (1.0 + np.exp(-self.direction * (features[:, 0] - self.midpoint)))
        return np.column_stack((1.0 - score, score))


def test_nested_loso_returns_per_subject_results() -> None:
    groups = np.repeat(np.asarray([1, 2, 3], dtype=np.int64), 8)
    labels = np.tile(np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64), 3)
    features = labels[:, None].astype(np.float64) * 4.0
    features += np.linspace(0, 0.2, len(features))[:, None]

    def factory(_labels: np.ndarray, _seed: int) -> _SimpleClassifier:
        return _SimpleClassifier()

    results = evaluate_loso(
        features,
        labels,
        groups,
        make_loso(groups),
        config=GBMConfig(inner_folds=2),
        model_factory=factory,
    )
    assert tuple(result.subject for result in results) == (1, 2, 3)
    assert all(0.05 <= result.threshold <= 0.95 for result in results)
    assert all(result.test_class_counts == {"non_fall": 4, "fall": 4} for result in results)
    aggregate = aggregate_subject_results(results)
    assert set(aggregate) == {
        "macro_f1",
        "fall_recall",
        "precision",
        "auprc",
        "false_positive_rate",
    }
