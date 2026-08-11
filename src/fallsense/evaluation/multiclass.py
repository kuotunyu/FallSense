"""11-class outer LOSO + inner grouped OOF calibration evaluation。"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
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


class MulticlassClassifier(Protocol):
    def fit(self, features: FloatArray, labels: IntArray) -> object: ...

    def predict_proba(self, features: FloatArray) -> FloatArray: ...


MulticlassModelFactory = Callable[[IntArray, int], MulticlassClassifier]


@dataclass(frozen=True)
class MulticlassConfig:
    num_classes: int = 11
    inner_folds: int = 3
    random_state: int = 20260719


@dataclass(frozen=True)
class MulticlassGBMConfig:
    n_estimators: int = 80
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 2.0
    n_jobs: int = 2
    random_state: int = 20260719
    inner_folds: int = 3
    num_classes: int = 11


@dataclass(frozen=True)
class ClassResult:
    support: int
    precision: float | None
    recall: float | None
    f1: float | None
    auprc: float | None


@dataclass(frozen=True)
class MulticlassSubjectResult:
    subject: int
    macro_f1: float
    present_classes: tuple[int, ...]
    train_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    train_class_weights: dict[str, float]
    per_class: dict[str, ClassResult]
    confusion_matrix: tuple[tuple[int, ...], ...]
    calibration: str


def balanced_class_weights(labels: IntArray, num_classes: int) -> FloatArray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if len(counts) != num_classes or np.any(counts == 0):
        raise ValueError("train fold 沒有覆蓋所有 11 classes")
    return (len(labels) / (num_classes * counts)).astype(np.float64)


class _WeightedXGBClassifier:
    def __init__(self, config: MulticlassGBMConfig, *, seed: int) -> None:
        module = importlib.import_module("xgboost")
        classifier = module.XGBClassifier
        self.num_classes = config.num_classes
        self.model = classifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            n_jobs=config.n_jobs,
            random_state=seed,
            objective="multi:softprob",
            num_class=config.num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
        )

    def fit(self, features: FloatArray, labels: IntArray) -> _WeightedXGBClassifier:
        class_weights = balanced_class_weights(labels, self.num_classes)
        self.model.fit(features, labels, sample_weight=class_weights[labels])
        return self

    def predict_proba(self, features: FloatArray) -> FloatArray:
        return cast(FloatArray, np.asarray(self.model.predict_proba(features), dtype=np.float64))


def multiclass_xgboost_factory(config: MulticlassGBMConfig) -> MulticlassModelFactory:
    def factory(_labels: IntArray, seed: int) -> _WeightedXGBClassifier:
        return _WeightedXGBClassifier(config, seed=seed)

    return factory


def _validated_probabilities(
    model: MulticlassClassifier,
    features: FloatArray,
    num_classes: int,
) -> FloatArray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError(f"predict_proba 必須輸出 [N,{num_classes}]")
    return probabilities


class _MultinomialCalibrator:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1_000,
            class_weight="balanced",
            random_state=0,
        )

    @staticmethod
    def _log_features(probabilities: FloatArray) -> FloatArray:
        return np.log(np.clip(probabilities, 1e-8, 1.0))

    def fit(self, probabilities: FloatArray, labels: IntArray) -> _MultinomialCalibrator:
        with warnings.catch_warnings():
            # Calibration 未收旂不能只留 warning 卻照常發佈 results。
            warnings.simplefilter("error", ConvergenceWarning)
            self.model.fit(self._log_features(probabilities), labels)
        if len(self.model.classes_) != self.num_classes:
            raise ValueError("calibration train fold 沒有覆蓋所有 classes")
        return self

    def predict(self, probabilities: FloatArray) -> FloatArray:
        return cast(
            FloatArray,
            np.asarray(
                self.model.predict_proba(self._log_features(probabilities)),
                dtype=np.float64,
            ),
        )


def _class_counts(labels: IntArray, num_classes: int) -> dict[str, int]:
    counts = np.bincount(labels, minlength=num_classes)
    return {f"activity_{index + 1}": int(counts[index]) for index in range(num_classes)}


def _per_class_results(
    labels: IntArray,
    predictions: IntArray,
    probabilities: FloatArray,
    num_classes: int,
) -> dict[str, ClassResult]:
    results: dict[str, ClassResult] = {}
    for class_index in range(num_classes):
        truth = labels == class_index
        predicted = predictions == class_index
        support = int(np.sum(truth))
        if support == 0:
            results[f"activity_{class_index + 1}"] = ClassResult(
                support=0,
                precision=None,
                recall=None,
                f1=None,
                auprc=None,
            )
            continue
        results[f"activity_{class_index + 1}"] = ClassResult(
            support=support,
            precision=float(precision_score(truth, predicted, zero_division=0)),
            recall=float(recall_score(truth, predicted, zero_division=0)),
            f1=float(f1_score(truth, predicted, zero_division=0)),
            auprc=float(average_precision_score(truth, probabilities[:, class_index])),
        )
    return results


def evaluate_multiclass_loso(
    features: FloatArray,
    labels: IntArray,
    groups: IntArray,
    folds: tuple[SubjectFold, ...],
    *,
    config: MulticlassConfig,
    model_factory: MulticlassModelFactory,
) -> tuple[MulticlassSubjectResult, ...]:
    """Held-out subject 絕不參與 model/calibration 訓練。"""

    all_classes = np.arange(config.num_classes, dtype=np.int64)
    results: list[MulticlassSubjectResult] = []
    for outer_number, fold in enumerate(folds):
        train_indices = np.asarray(fold.train_indices, dtype=np.int64)
        test_indices = np.asarray(fold.test_indices, dtype=np.int64)
        train_x, train_y, train_groups = (
            features[train_indices],
            labels[train_indices],
            groups[train_indices],
        )
        test_x, test_y = features[test_indices], labels[test_indices]
        inner_splits = min(config.inner_folds, len(np.unique(train_groups)))
        splitter = GroupKFold(n_splits=inner_splits)
        oof = np.full((len(train_y), config.num_classes), np.nan, dtype=np.float64)
        for inner_number, (inner_train, inner_valid) in enumerate(
            splitter.split(train_x, train_y, groups=train_groups)
        ):
            model = model_factory(
                train_y[inner_train],
                config.random_state + outer_number * 100 + inner_number,
            )
            model.fit(train_x[inner_train], train_y[inner_train])
            oof[inner_valid] = _validated_probabilities(
                model,
                train_x[inner_valid],
                config.num_classes,
            )
        if np.isnan(oof).any():
            raise AssertionError("inner grouped OOF probabilities 不完整")
        calibrator = _MultinomialCalibrator(config.num_classes).fit(oof, train_y)
        final_model = model_factory(train_y, config.random_state + outer_number * 1_000)
        final_model.fit(train_x, train_y)
        calibrated = calibrator.predict(
            _validated_probabilities(final_model, test_x, config.num_classes)
        )
        prediction = np.argmax(calibrated, axis=1).astype(np.int64)
        present_classes = np.unique(test_y)
        matrix = confusion_matrix(test_y, prediction, labels=all_classes).astype(int)
        class_weights = balanced_class_weights(train_y, config.num_classes)
        results.append(
            MulticlassSubjectResult(
                subject=fold.test_subjects[0],
                macro_f1=float(
                    f1_score(
                        test_y,
                        prediction,
                        labels=present_classes,
                        average="macro",
                        zero_division=0,
                    )
                ),
                present_classes=tuple(int(value + 1) for value in present_classes),
                train_class_counts=_class_counts(train_y, config.num_classes),
                test_class_counts=_class_counts(test_y, config.num_classes),
                train_class_weights={
                    f"activity_{index + 1}": float(class_weights[index])
                    for index in range(config.num_classes)
                },
                per_class=_per_class_results(
                    test_y,
                    prediction,
                    calibrated,
                    config.num_classes,
                ),
                confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
                calibration="multinomial logistic on outer-train grouped OOF probabilities",
            )
        )
    return tuple(results)


def aggregate_multiclass_results(
    results: tuple[MulticlassSubjectResult, ...],
    num_classes: int,
) -> dict[str, object]:
    macro = [result.macro_f1 for result in results]
    per_class: dict[str, dict[str, dict[str, float] | int]] = {}
    for class_index in range(num_classes):
        name = f"activity_{class_index + 1}"
        class_payload: dict[str, dict[str, float] | int] = {}
        available_subjects = 0
        for metric in ("precision", "recall", "f1", "auprc"):
            values = [
                value
                for result in results
                if (value := getattr(result.per_class[name], metric)) is not None
            ]
            available_subjects = max(available_subjects, len(values))
            class_payload[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        class_payload["subjects_with_support"] = available_subjects
        per_class[name] = class_payload
    return {
        "macro_f1": {
            "mean": float(np.mean(macro)),
            "std": float(np.std(macro, ddof=1)),
        },
        "per_class": per_class,
    }


def multiclass_subject_payload(
    results: tuple[MulticlassSubjectResult, ...],
) -> list[dict[str, object]]:
    return [asdict(result) for result in results]
