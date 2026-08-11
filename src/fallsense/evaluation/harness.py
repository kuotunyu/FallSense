"""Outer LOSO + inner grouped OOF calibration/threshold evaluation harness。"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

from fallsense.evaluation.splits import SubjectFold

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ProbabilisticClassifier(Protocol):
    def fit(self, features: FloatArray, labels: IntArray) -> object: ...

    def predict_proba(self, features: FloatArray) -> FloatArray: ...


ModelFactory = Callable[[IntArray, int], ProbabilisticClassifier]


@dataclass(frozen=True)
class GBMConfig:
    n_estimators: int = 120
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 2.0
    n_jobs: int = 2
    random_state: int = 20260719
    inner_folds: int = 5
    calibration: str = "sigmoid"


@dataclass(frozen=True)
class SigmoidCalibrator:
    coefficient: float
    intercept: float

    def predict(self, probabilities: FloatArray) -> FloatArray:
        logits = np.clip(self.coefficient * probabilities + self.intercept, -709.0, 709.0)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    macro_f1: float
    fall_recall: float
    precision: float
    false_positive_rate: float


@dataclass(frozen=True)
class SubjectResult:
    subject: int
    threshold: float
    macro_f1: float
    fall_recall: float
    precision: float
    auprc: float
    false_positive_rate: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    train_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    scale_pos_weight: float
    calibration_coefficient: float
    calibration_intercept: float
    threshold_curve: tuple[ThresholdPoint, ...]


def _class_counts(labels: IntArray) -> dict[str, int]:
    return {
        "non_fall": int(np.sum(labels == 0)),
        "fall": int(np.sum(labels == 1)),
    }


def _scale_pos_weight(labels: IntArray) -> float:
    counts = _class_counts(labels)
    if counts["fall"] == 0:
        raise ValueError("train fold 沒有 fall samples")
    return counts["non_fall"] / counts["fall"]


def xgboost_factory(config: GBMConfig) -> ModelFactory:
    """只在執行 training 時 import optional xgboost。"""

    module = importlib.import_module("xgboost")
    classifier = module.XGBClassifier

    def factory(labels: IntArray, seed: int) -> ProbabilisticClassifier:
        return cast(
            ProbabilisticClassifier,
            classifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                min_child_weight=config.min_child_weight,
                n_jobs=config.n_jobs,
                random_state=seed,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=_scale_pos_weight(labels),
            ),
        )

    return factory


def fit_sigmoid_calibrator(probabilities: FloatArray, labels: IntArray) -> SigmoidCalibrator:
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", random_state=0)
    model.fit(probabilities.reshape(-1, 1), labels)
    return SigmoidCalibrator(float(model.coef_[0, 0]), float(model.intercept_[0]))


def _fpr(labels: IntArray, predictions: IntArray) -> float:
    tn, fp, _, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if fp + tn else 0.0


def threshold_curve(labels: IntArray, probabilities: FloatArray) -> tuple[ThresholdPoint, ...]:
    points: list[ThresholdPoint] = []
    for threshold in np.linspace(0.05, 0.95, 19):
        predictions = (probabilities >= threshold).astype(np.int64)
        points.append(
            ThresholdPoint(
                threshold=float(threshold),
                macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
                fall_recall=float(recall_score(labels, predictions, zero_division=0)),
                precision=float(precision_score(labels, predictions, zero_division=0)),
                false_positive_rate=_fpr(labels, predictions),
            )
        )
    return tuple(points)


def choose_threshold(points: tuple[ThresholdPoint, ...]) -> ThresholdPoint:
    """只在 outer-train OOF 上，以 macro-F1、recall、低 FPR 依序決定 threshold。"""

    return max(
        points, key=lambda point: (point.macro_f1, point.fall_recall, -point.false_positive_rate)
    )


def _positive_probability(model: ProbabilisticClassifier, features: FloatArray) -> FloatArray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("classifier.predict_proba 必須輸出 [N, 2]")
    return probabilities[:, 1]


def evaluate_loso(
    features: FloatArray,
    labels: IntArray,
    groups: IntArray,
    folds: tuple[SubjectFold, ...],
    *,
    config: GBMConfig,
    model_factory: ModelFactory | None = None,
) -> tuple[SubjectResult, ...]:
    """每個 outer held-out subject 都獨立做 inner OOF calibration 與 threshold selection。"""

    factory = model_factory or xgboost_factory(config)
    results: list[SubjectResult] = []
    for outer_number, fold in enumerate(folds):
        train_indices = np.asarray(fold.train_indices, dtype=np.int64)
        test_indices = np.asarray(fold.test_indices, dtype=np.int64)
        train_x, train_y, train_groups = (
            features[train_indices],
            labels[train_indices],
            groups[train_indices],
        )
        test_x, test_y = features[test_indices], labels[test_indices]
        unique_train_groups = np.unique(train_groups)
        inner_splits = min(config.inner_folds, len(unique_train_groups))
        if inner_splits < 2:
            raise ValueError("outer train 至少需要兩個 subjects 才能做 grouped calibration")
        splitter = GroupKFold(n_splits=inner_splits)
        oof = np.full(len(train_y), np.nan, dtype=np.float64)
        for inner_number, (inner_train, inner_valid) in enumerate(
            splitter.split(train_x, train_y, groups=train_groups)
        ):
            seed = config.random_state + outer_number * 100 + inner_number
            model = factory(train_y[inner_train], seed)
            model.fit(train_x[inner_train], train_y[inner_train])
            oof[inner_valid] = _positive_probability(model, train_x[inner_valid])
        if np.isnan(oof).any():
            raise AssertionError("inner grouped OOF probabilities 不完整")
        calibrator = fit_sigmoid_calibrator(oof, train_y)
        calibrated_oof = calibrator.predict(oof)
        curve = threshold_curve(train_y, calibrated_oof)
        selected = choose_threshold(curve)

        final_model = factory(train_y, config.random_state + outer_number * 1000)
        final_model.fit(train_x, train_y)
        test_probability = calibrator.predict(_positive_probability(final_model, test_x))
        prediction = (test_probability >= selected.threshold).astype(np.int64)
        confusion = confusion_matrix(test_y, prediction, labels=[0, 1]).astype(int)
        results.append(
            SubjectResult(
                subject=fold.test_subjects[0],
                threshold=selected.threshold,
                macro_f1=float(f1_score(test_y, prediction, average="macro", zero_division=0)),
                fall_recall=float(recall_score(test_y, prediction, zero_division=0)),
                precision=float(precision_score(test_y, prediction, zero_division=0)),
                auprc=float(average_precision_score(test_y, test_probability)),
                false_positive_rate=_fpr(test_y, prediction),
                confusion_matrix=(
                    (int(confusion[0, 0]), int(confusion[0, 1])),
                    (int(confusion[1, 0]), int(confusion[1, 1])),
                ),
                train_class_counts=_class_counts(train_y),
                test_class_counts=_class_counts(test_y),
                scale_pos_weight=_scale_pos_weight(train_y),
                calibration_coefficient=calibrator.coefficient,
                calibration_intercept=calibrator.intercept,
                threshold_curve=curve,
            )
        )
    return tuple(results)


def aggregate_subject_results(results: tuple[SubjectResult, ...]) -> dict[str, dict[str, float]]:
    if not results:
        raise ValueError("沒有 subject results")
    metrics = ("macro_f1", "fall_recall", "precision", "auprc", "false_positive_rate")
    return {
        metric: {
            "mean": float(np.mean([getattr(result, metric) for result in results])),
            "std": float(np.std([getattr(result, metric) for result in results], ddof=1)),
        }
        for metric in metrics
    }


def subject_results_payload(results: tuple[SubjectResult, ...]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]
