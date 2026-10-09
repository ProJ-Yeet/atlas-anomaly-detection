"""Precision-Recall curve plotting (importable helper + standalone CLI).

PR curves are the honest view for a ~5% base rate: unlike ROC they are not
inflated by the abundant negatives, so PR-AUC (average precision) is the metric
to trust on this imbalanced problem.

Standalone::

    python evaluation/precision_recall.py figures/isolation_forest_predictions.csv
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402


def plot_pr(ax, y_true: np.ndarray, scores: np.ndarray,
            label: str = "model") -> float:
    """Plot the PR curve on ``ax`` and return the average precision (PR-AUC)."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    ax.plot(recall, precision, lw=1.5, label=f"{label} (AP={ap:.3f})")
    base = np.mean(y_true)
    ax.axhline(base, ls="--", color="grey", lw=0.8,
               label=f"base rate={base:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right", fontsize=8)
    return float(ap)


def _main(csv_path: str) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_pr(ax, df["anomaly"].to_numpy(), df["anomaly_score"].to_numpy(),
            label=os.path.basename(csv_path).split("_predictions")[0])
    out = csv_path.replace("_predictions.csv", "_pr.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python evaluation/precision_recall.py <predictions.csv>")
        raise SystemExit(1)
    _main(sys.argv[1])
