"""Standard figures shared by every detector.

Uses a non-interactive Agg backend so scripts render to PNG without a display
(headless servers, CI, Windows batch runs).  Every algorithm produces the same
multi-panel "result figure" so the methods can be compared visually side by
side; individual helpers are also exposed for bespoke plots.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_PKG_ROOT, "figures")


def _ensure_dir() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> str:
    """Save ``fig`` as ``figures/<name>.png`` and close it. Returns the path."""
    _ensure_dir()
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Individual panels
# --------------------------------------------------------------------------- #

def plot_timeseries(ax, series: np.ndarray, y_true: Optional[np.ndarray] = None,
                    title: str = "temperature", ylabel: str = "value") -> None:
    """Line plot of one channel with true-anomaly regions shaded."""
    ax.plot(series, lw=0.7, color="#2c6fbb")
    if y_true is not None:
        ax.fill_between(np.arange(len(series)), series.min(), series.max(),
                        where=np.asarray(y_true).astype(bool),
                        color="#e15759", alpha=0.25, label="true anomaly")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)


def plot_scores(ax, scores: np.ndarray, threshold: float,
                y_true: Optional[np.ndarray] = None,
                title: str = "anomaly score") -> None:
    """Anomaly score with the decision threshold and flagged points."""
    ax.plot(scores, lw=0.7, color="#444444")
    ax.axhline(threshold, color="#e15759", ls="--", lw=1.0,
               label=f"threshold={threshold:.3f}")
    pred = np.asarray(scores) >= threshold
    ax.scatter(np.where(pred)[0], np.asarray(scores)[pred], s=6,
               color="#e15759", label="flagged", zorder=3)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)


def plot_score_hist(ax, scores: np.ndarray, y_true: np.ndarray,
                    threshold: Optional[float] = None) -> None:
    """Histogram of scores split by true class -- shows tail separation."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true).astype(bool)
    bins = np.linspace(np.min(scores), np.quantile(scores, 0.999), 60)
    ax.hist(scores[~y_true], bins=bins, alpha=0.6, color="#2c6fbb",
            label="normal", density=True)
    ax.hist(scores[y_true], bins=bins, alpha=0.6, color="#e15759",
            label="anomaly", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=1.0)
    ax.set_title("score histogram", fontsize=10)
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)


def plot_pca(ax, X: np.ndarray, labels: np.ndarray,
             title: str = "PCA projection", max_points: int = 6000) -> None:
    """2-D PCA scatter coloured by ``labels`` (0/1 or cluster id).

    Subsamples to ``max_points`` (keeping all anomalies when labels are binary)
    so a 100k-row scatter stays fast and legible.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if len(X) > max_points:
        rng = np.random.default_rng(0)
        pos = np.where(labels == 1)[0]
        neg = np.where(labels != 1)[0]
        n_pos = min(len(pos), max_points // 2)
        n_neg = min(len(neg), max_points - n_pos)
        pos_keep = rng.choice(pos, n_pos, replace=False) if len(pos) else pos
        neg_keep = rng.choice(neg, n_neg, replace=False) if len(neg) else neg
        idx = np.concatenate([pos_keep, neg_keep]).astype(int)
        X, labels = X[idx], labels[idx]
    if X.shape[1] > 2:
        X2 = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        X2 = X
    for lab in np.unique(labels):
        m = labels == lab
        ax.scatter(X2[m, 0], X2[m, 1], s=4, alpha=0.5, label=str(lab))
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, markerscale=2)


def plot_feature_importance(ax, names: Sequence[str], importances: np.ndarray,
                            top: int = 15) -> None:
    """Horizontal bar chart of the ``top`` most important features."""
    importances = np.asarray(importances, dtype=float)
    order = np.argsort(importances)[::-1][:top][::-1]
    ax.barh(range(len(order)), importances[order], color="#59a14f")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_title("feature importance", fontsize=10)


# --------------------------------------------------------------------------- #
# Combined result figure
# --------------------------------------------------------------------------- #

def result_figure(name: str, channel: np.ndarray, scores: np.ndarray,
                  threshold: float, y_true: np.ndarray,
                  X_for_pca: Optional[np.ndarray] = None,
                  pca_labels: Optional[np.ndarray] = None,
                  channel_name: str = "temperature",
                  extra_panel=None) -> str:
    """Render the standard 4-panel figure and save it under ``figures/``.

    Panels: (1) channel with truth shaded, (2) score + threshold + flags,
    (3) score histogram by class, (4) PCA scatter coloured by prediction
    (or a caller-supplied ``extra_panel(ax)`` diagnostic).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{name}", fontsize=13, fontweight="bold")

    plot_timeseries(axes[0, 0], channel, y_true, title=channel_name,
                    ylabel=channel_name)
    plot_scores(axes[0, 1], scores, threshold, y_true)
    plot_score_hist(axes[1, 0], scores, y_true, threshold)

    if extra_panel is not None:
        extra_panel(axes[1, 1])
    elif X_for_pca is not None:
        lab = pca_labels if pca_labels is not None else (scores >= threshold)
        plot_pca(axes[1, 1], X_for_pca, lab.astype(int),
                 title="PCA (coloured by prediction)")
    else:
        axes[1, 1].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return savefig(fig, name)


def full_report_figure(name: str, scores: np.ndarray, threshold: float,
                       y_true: np.ndarray, X_for_pca: np.ndarray,
                       ts_channel: np.ndarray, ts_scores: np.ndarray,
                       ts_y: np.ndarray,
                       feature_names: Optional[Sequence[str]] = None,
                       feature_importance: Optional[np.ndarray] = None,
                       cluster_labels: Optional[np.ndarray] = None,
                       channel_name: str = "temperature") -> str:
    """Render the full 3x3 diagnostic panel required by the brief.

    The metric panels (histogram / ROC / PR / confusion / PCA) use the
    **global** ``scores`` / ``y_true`` / ``X_for_pca``; the two top-left
    time-series panels use the single-element ``ts_channel`` / ``ts_scores`` /
    ``ts_y`` so one incident is legible on the time axis.

    Panels: time series w/ anomaly overlay | score + threshold + flags | score
    histogram || ROC | PR | confusion matrix || PCA coloured by prediction |
    PCA/cluster coloured by truth (or ``cluster_labels``) | feature importance.

    Falls back gracefully when ``feature_importance`` is absent (unsupervised
    models) -- that panel is turned off.
    """
    from evaluation.roc_curve import plot_roc
    from evaluation.precision_recall import plot_pr
    from evaluation.confusion_matrix import plot_confusion

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(name, fontsize=14, fontweight="bold")

    plot_timeseries(ax[0, 0], ts_channel, ts_y, title=channel_name,
                    ylabel=channel_name)
    plot_scores(ax[0, 1], ts_scores, threshold, ts_y)
    plot_score_hist(ax[0, 2], scores, y_true, threshold)

    # ROC / PR need both classes present.
    if len(np.unique(y_true)) > 1:
        plot_roc(ax[1, 0], y_true, scores, label=name)
        plot_pr(ax[1, 1], y_true, scores, label=name)
    else:
        ax[1, 0].axis("off"); ax[1, 1].axis("off")
    plot_confusion(ax[1, 2], y_true, y_pred)

    plot_pca(ax[2, 0], X_for_pca, y_pred, title="PCA (predicted)")
    lab = cluster_labels if cluster_labels is not None else y_true
    plot_pca(ax[2, 1], X_for_pca, np.asarray(lab).astype(int),
             title="PCA (truth / clusters)")
    if feature_importance is not None and feature_names is not None:
        plot_feature_importance(ax[2, 2], feature_names, feature_importance)
    else:
        ax[2, 2].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return savefig(fig, name)
