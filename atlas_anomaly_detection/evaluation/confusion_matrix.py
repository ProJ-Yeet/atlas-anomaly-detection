"""Confusion-matrix plotting (importable helper + standalone CLI).

Standalone::

    python evaluation/confusion_matrix.py figures/isolation_forest_predictions.csv
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def plot_confusion(ax, y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = "confusion matrix") -> np.ndarray:
    """Plot a 2x2 confusion matrix on ``ax`` and return the counts."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "anomaly"])
    ax.set_yticklabels(["normal", "anomaly"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=11)
    return cm


def _main(csv_path: str) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_confusion(ax, df["anomaly"].to_numpy(), df["predicted"].to_numpy(),
                   title=os.path.basename(csv_path).split("_predictions")[0])
    out = csv_path.replace("_predictions.csv", "_cm.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python evaluation/confusion_matrix.py <predictions.csv>")
        raise SystemExit(1)
    _main(sys.argv[1])
