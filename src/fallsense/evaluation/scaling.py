"""只能以 train fold fit 的 per-channel standardizer。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FoldStandardizer:
    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, train_values: FloatArray) -> FoldStandardizer:
        if train_values.ndim != 3 or train_values.shape[0] == 0:
            raise ValueError("train_values 必須是非空 [windows, time, channels]")
        mean = np.nanmean(train_values, axis=(0, 1))
        scale = np.nanstd(train_values, axis=(0, 1))
        mean = np.where(np.isnan(mean), 0.0, mean)
        scale = np.where(np.isnan(scale) | (scale == 0.0), 1.0, scale)
        return cls(mean.astype(np.float64), scale.astype(np.float64))

    def transform(self, values: FloatArray) -> FloatArray:
        return ((values - self.mean) / self.scale).astype(np.float64)


def fit_on_train_indices(values: FloatArray, train_indices: tuple[int, ...]) -> FoldStandardizer:
    """介面明確要求 train indices，避免呼叫端方便地 fit 全資料。"""

    return FoldStandardizer.fit(values[np.asarray(train_indices, dtype=np.int64)])
