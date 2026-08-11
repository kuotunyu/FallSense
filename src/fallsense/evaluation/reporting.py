"""從 results payload 產生 per-subject Markdown、threshold curve 與 confusion matrix。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np


def write_markdown_report(payload: dict[str, object], path: Path) -> None:
    aggregate = cast(dict[str, dict[str, float]], payload["aggregate"])
    per_subject = cast(list[dict[str, Any]], payload["per_subject"])
    lines = [
        f"# {payload['run_id']} — binary LOSO report",
        "",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload['git_commit']}`",
        f"- split checksum: `{payload['split_checksum']}`",
        "- threshold: 每個 outer fold 只以 outer-train OOF calibrated probabilities 選擇",
        "",
        "## Headline metrics（per-held-out-subject mean±std）",
        "",
        "| metric | mean±std |",
        "|---|---:|",
    ]
    labels = {
        "macro_f1": "macro-F1",
        "fall_recall": "fall recall",
        "precision": "precision",
        "auprc": "AUPRC",
        "false_positive_rate": "false-positive rate",
    }
    for key, label in labels.items():
        value = aggregate[key]
        lines.append(f"| {label} | {value['mean']:.4f}±{value['std']:.4f} |")
    lines.extend(
        [
            "",
            "## Per-subject",
            "",
            "| subject | threshold | macro-F1 | fall recall | precision | AUPRC | FPR |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_subject:
        lines.append(
            "| {subject} | {threshold:.2f} | {macro_f1:.4f} | {fall_recall:.4f} | "
            "{precision:.4f} | {auprc:.4f} | {false_positive_rate:.4f} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def plot_report_figures(payload: dict[str, object], output_dir: Path) -> None:
    per_subject = cast(list[dict[str, Any]], payload["per_subject"])
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.sum([np.asarray(row["confusion_matrix"], dtype=int) for row in per_subject], axis=0)
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for matrix_row in range(2):
        for column in range(2):
            axis.text(
                column,
                matrix_row,
                str(matrix[matrix_row, column]),
                ha="center",
                va="center",
            )
    axis.set_xticks((0, 1), ("non-fall", "fall"))
    axis.set_yticks((0, 1), ("non-fall", "fall"))
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title("LOSO confusion matrix (sum of subjects)")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for subject_row in per_subject:
        curve = subject_row["threshold_curve"]
        axis.plot(
            [point["threshold"] for point in curve],
            [point["macro_f1"] for point in curve],
            alpha=0.35,
        )
    axis.set_xlabel("threshold")
    axis.set_ylabel("outer-train OOF macro-F1")
    axis.set_title("Per-subject threshold sweeps (selection data only)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "threshold_sweep.png", dpi=150)
    plt.close(figure)
