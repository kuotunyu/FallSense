"""感測器與 row-level Tag 對齊圖。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fallsense.data.schema import TAG_COLUMN, TIMESTAMP_COLUMN


def plot_trial_alignment(frame: pd.DataFrame, output: Path, *, title: str) -> None:
    time = frame[TIMESTAMP_COLUMN].to_numpy(dtype=np.float64)
    ankle = np.linalg.norm(
        frame[["LeftAnkle_AccX", "LeftAnkle_AccY", "LeftAnkle_AccZ"]].to_numpy(dtype=float),
        axis=1,
    )
    waist = np.linalg.norm(
        frame[["Waist_AccX", "Waist_AccY", "Waist_AccZ"]].to_numpy(dtype=float),
        axis=1,
    )
    tags = frame[TAG_COLUMN].to_numpy(dtype=np.int64)
    figure, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, height_ratios=(3, 1))
    axes[0].plot(time, ankle, label="left ankle |acc|", linewidth=1.2)
    axes[0].plot(time, waist, label="waist |acc|", linewidth=1.2)
    axes[0].set_ylabel("acceleration magnitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)
    axes[1].step(time, tags, where="post", color="#c2410c")
    axes[1].set_ylabel("Tag")
    axes[1].set_xlabel("seconds")
    axes[1].grid(alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
