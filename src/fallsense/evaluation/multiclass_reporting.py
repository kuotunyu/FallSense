"""11-class per-subject / per-class report and confusion matrix。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np


def write_multiclass_report(payload: dict[str, object], path: Path) -> None:
    aggregate = cast(dict[str, Any], payload["aggregate"])
    per_subject = cast(list[dict[str, Any]], payload["per_subject"])
    macro = aggregate["macro_f1"]
    lines = [
        f"# {payload['run_id']} — 11-class LOSO report",
        "",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload['git_commit']}`",
        f"- split checksum: `{payload['split_checksum']}`",
        "- calibration: multinomial logistic regression fitted only on outer-train "
        "grouped OOF probabilities",
        "- per-subject macro-F1 averages only classes present in that held-out subject; "
        "missing-support classes are reported as unavailable",
        "",
        "## Headline metric (per-held-out-subject mean±std)",
        "",
        f"- macro-F1: **{macro['mean']:.4f}±{macro['std']:.4f}**",
        "",
        "## Per-subject",
        "",
        "| subject | macro-F1 | classes with support |",
        "|---:|---:|---:|",
    ]
    for row in per_subject:
        lines.append(
            f"| {row['subject']} | {row['macro_f1']:.4f} | {len(row['present_classes'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-class (mean±std over held-out subjects with support)",
            "",
            "| class | subjects | precision | recall | F1 | AUPRC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in aggregate["per_class"].items():
        lines.append(
            f"| {name} | {row['subjects_with_support']} | "
            f"{row['precision']['mean']:.4f}±{row['precision']['std']:.4f} | "
            f"{row['recall']['mean']:.4f}±{row['recall']['std']:.4f} | "
            f"{row['f1']['mean']:.4f}±{row['f1']['std']:.4f} | "
            f"{row['auprc']['mean']:.4f}±{row['auprc']['std']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def plot_multiclass_confusion(payload: dict[str, object], output: Path) -> None:
    per_subject = cast(list[dict[str, Any]], payload["per_subject"])
    matrix = np.sum([np.asarray(row["confusion_matrix"], dtype=int) for row in per_subject], axis=0)
    row_total = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_total,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_total > 0,
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = np.arange(11)
    axis.set_xticks(ticks, [str(index) for index in range(1, 12)])
    axis.set_yticks(ticks, [str(index) for index in range(1, 12)])
    axis.set_xlabel("predicted activity")
    axis.set_ylabel("true activity")
    axis.set_title("11-class LOSO row-normalized confusion matrix")
    figure.colorbar(image, ax=axis, label="row proportion")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
