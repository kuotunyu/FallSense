from __future__ import annotations

import numpy as np

from fallsense.evaluation.multiclass import (
    MulticlassConfig,
    aggregate_multiclass_results,
    balanced_class_weights,
    evaluate_multiclass_loso,
)
from fallsense.evaluation.splits import make_loso


class _CentroidClassifier:
    def fit(self, features: np.ndarray, labels: np.ndarray) -> object:
        self.centroids = np.asarray(
            [np.mean(features[labels == class_index, 0]) for class_index in range(11)]
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        distance = np.abs(features[:, :1] - self.centroids.reshape(1, -1))
        score = np.exp(-distance)
        return score / score.sum(axis=1, keepdims=True)


def test_balanced_weights_are_fit_from_supplied_labels() -> None:
    labels = np.repeat(np.arange(11, dtype=np.int64), np.arange(1, 12))
    weights = balanced_class_weights(labels, 11)
    weighted_counts = np.bincount(labels, weights=weights[labels], minlength=11)
    np.testing.assert_allclose(weighted_counts, np.repeat(weighted_counts[0], 11))


def test_multiclass_loso_reports_each_subject_and_class() -> None:
    labels = np.tile(np.repeat(np.arange(11, dtype=np.int64), 2), 3)
    groups = np.repeat(np.arange(1, 4, dtype=np.int64), 22)
    features = labels[:, None].astype(np.float64)

    def factory(_labels: np.ndarray, _seed: int) -> _CentroidClassifier:
        return _CentroidClassifier()

    results = evaluate_multiclass_loso(
        features,
        labels,
        groups,
        make_loso(groups),
        config=MulticlassConfig(inner_folds=2),
        model_factory=factory,
    )
    assert [result.subject for result in results] == [1, 2, 3]
    assert all(len(result.per_class) == 11 for result in results)
    aggregate = aggregate_multiclass_results(results, 11)
    assert set(aggregate) == {"macro_f1", "per_class"}
