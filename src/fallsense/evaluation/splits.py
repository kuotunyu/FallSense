"""LOSO / GroupKFold split 與 deterministic manifest。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SubjectFold:
    fold_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]


def _fold(
    fold_id: str, train: NDArray[np.int64], test: NDArray[np.int64], groups: IntArray
) -> SubjectFold:
    train_subjects = tuple(sorted(int(value) for value in np.unique(groups[train])))
    test_subjects = tuple(sorted(int(value) for value in np.unique(groups[test])))
    if set(train_subjects) & set(test_subjects):
        raise AssertionError(f"{fold_id} 發生 subject overlap")
    return SubjectFold(
        fold_id,
        tuple(int(value) for value in train),
        tuple(int(value) for value in test),
        train_subjects,
        test_subjects,
    )


def make_loso(groups: IntArray) -> tuple[SubjectFold, ...]:
    indices = np.arange(len(groups), dtype=np.int64)
    splitter = LeaveOneGroupOut()
    return tuple(
        _fold(f"loso-subject-{int(np.unique(groups[test])[0])}", train, test, groups)
        for train, test in splitter.split(indices, groups=groups)
    )


def make_group_kfold(groups: IntArray, *, n_splits: int = 5) -> tuple[SubjectFold, ...]:
    indices = np.arange(len(groups), dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits)
    return tuple(
        _fold(f"group-kfold-{number}", train, test, groups)
        for number, (train, test) in enumerate(splitter.split(indices, groups=groups), start=1)
    )


def split_manifest_payload(
    folds: tuple[SubjectFold, ...],
    window_ids: tuple[str, ...],
    *,
    config_checksum: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "config_checksum": config_checksum,
        "window_ids": list(window_ids),
        "folds": [asdict(fold) for fold in folds],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["checksum"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def save_split_manifest(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
