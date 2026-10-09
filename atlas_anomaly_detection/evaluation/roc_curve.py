"""ROC curve plotting (importable helper + standalone CLI).

As a module::

    from evaluation.roc_curve import plot_roc

As a script (plots the ROC of a saved predictions CSV)::

    python evaluation/roc_curve.py figures/isolation_forest_predictions.csv
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402


def plot_roc(ax, y_true: np.ndarray, scores: np.ndarray,
             label: str = "model") -> float:
    """Plot the ROC curve on ``ax`` and return the AUC."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    ax.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    return float(auc)


def _main(csv_path: str) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_roc(ax, df["anomaly"].to_numpy(), df["anomaly_score"].to_numpy(),
             label=os.path.basename(csv_path).split("_predictions")[0])
    out = csv_path.replace("_predictions.csv", "_roc.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python evaluation/roc_curve.py <predictions.csv>")
        raise SystemExit(1)
    _main(sys.argv[1])
