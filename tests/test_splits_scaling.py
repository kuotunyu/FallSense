from __future__ import annotations

import numpy as np

from fallsense.evaluation.scaling import fit_on_train_indices
from fallsense.evaluation.splits import make_group_kfold, make_loso, split_manifest_payload


def test_loso_and_group_kfold_have_no_subject_overlap() -> None:
    groups = np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
    folds = (*make_loso(groups), *make_group_kfold(groups, n_splits=5))
    for fold in folds:
        assert set(fold.train_subjects).isdisjoint(fold.test_subjects)
        assert set(fold.train_indices).isdisjoint(fold.test_indices)


def test_split_manifest_checksum_is_reproducible() -> None:
    groups = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
    folds = make_loso(groups)
    ids = tuple(f"window-{index}" for index in range(len(groups)))
    first = split_manifest_payload(folds, ids, config_checksum="config")
    second = split_manifest_payload(folds, ids, config_checksum="config")
    assert first == second


def test_scaler_ignores_held_out_subject_and_differs_between_folds() -> None:
    groups = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
    values = np.zeros((6, 4, 2), dtype=np.float64)
    values[groups == 1] = 1.0
    values[groups == 2] = 10.0
    values[groups == 3] = 100.0
    folds = make_loso(groups)
    held_out_three = next(fold for fold in folds if fold.test_subjects == (3,))
    scaler = fit_on_train_indices(values, held_out_three.train_indices)

    perturbed_test = values.copy()
    perturbed_test[np.asarray(held_out_three.test_indices)] = 999_999.0
    same_scaler = fit_on_train_indices(perturbed_test, held_out_three.train_indices)
    np.testing.assert_array_equal(scaler.mean, same_scaler.mean)
    np.testing.assert_array_equal(scaler.scale, same_scaler.scale)

    held_out_one = next(fold for fold in folds if fold.test_subjects == (1,))
    other_scaler = fit_on_train_indices(values, held_out_one.train_indices)
    assert not np.array_equal(scaler.mean, other_scaler.mean)
