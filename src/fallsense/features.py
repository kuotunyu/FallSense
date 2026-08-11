"""Window-level handcrafted time/frequency features。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fallsense.data.schema import SENSOR_COLUMNS

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FeatureMatrix:
    values: FloatArray
    names: tuple[str, ...]


STAT_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "rms",
    "absmax",
    "mean_abs_diff",
    "dominant_frequency_bin",
    "spectral_centroid_bin",
)


def extract_handcrafted_features(windows: FloatArray) -> FeatureMatrix:
    """輸入 [windows, time, channels]，缺整個 channel 時保留 NaN 供 GBM 處理。"""

    if windows.ndim != 3 or windows.shape[2] != len(SENSOR_COLUMNS):
        raise ValueError("windows 必須是 [N, time, 42 channels]")
    missing = np.all(np.isnan(windows), axis=1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(windows, axis=1)
        std = np.nanstd(windows, axis=1)
        minimum = np.nanmin(windows, axis=1)
        maximum = np.nanmax(windows, axis=1)
        median = np.nanmedian(windows, axis=1)
        q25 = np.nanpercentile(windows, 25, axis=1)
        q75 = np.nanpercentile(windows, 75, axis=1)
        rms = np.sqrt(np.nanmean(np.square(windows), axis=1))
        absmax = np.nanmax(np.abs(windows), axis=1)
        mean_abs_diff = np.nanmean(np.abs(np.diff(windows, axis=1)), axis=1)

    filled = np.where(np.isnan(windows), median[:, None, :], windows)
    filled = np.nan_to_num(filled, nan=0.0)
    spectrum = np.abs(np.fft.rfft(filled, axis=1)) ** 2
    non_dc = spectrum[:, 1:, :]
    bins = np.arange(1, spectrum.shape[1], dtype=np.float64)[None, :, None]
    dominant = np.argmax(non_dc, axis=1).astype(np.float64) + 1.0
    denominator = np.sum(non_dc, axis=1)
    centroid = np.divide(
        np.sum(non_dc * bins, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    features_by_stat = (
        mean,
        std,
        minimum,
        maximum,
        median,
        q75 - q25,
        rms,
        absmax,
        mean_abs_diff,
        dominant,
        centroid,
    )
    cleaned = [np.where(missing, np.nan, feature) for feature in features_by_stat]
    matrix = np.concatenate(cleaned, axis=1).astype(np.float64)
    names = tuple(f"{stat}__{channel}" for stat in STAT_NAMES for channel in SENSOR_COLUMNS)
    return FeatureMatrix(matrix, names)


def select_feature_channels(
    features: FeatureMatrix,
    channels: tuple[str, ...],
) -> FeatureMatrix:
    """依原始 sensor channel 選擇每一組 statistics features。"""

    unknown = sorted(set(channels) - set(SENSOR_COLUMNS))
    if unknown:
        raise ValueError(f"未知 sensor channels：{unknown}")
    if not channels:
        raise ValueError("至少需要一個 sensor channel")
    channel_set = set(channels)
    indices = tuple(
        index
        for index, name in enumerate(features.names)
        if name.split("__", maxsplit=1)[1] in channel_set
    )
    expected = len(STAT_NAMES) * len(channels)
    if len(indices) != expected:
        raise AssertionError(f"預期 {expected} features，實際選到 {len(indices)}")
    return FeatureMatrix(
        values=features.values[:, indices],
        names=tuple(features.names[index] for index in indices),
    )
