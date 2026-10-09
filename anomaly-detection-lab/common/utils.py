"""Shared helpers: data loading, windowing, evaluation, result plotting."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data_generator import CHANNELS, DATA_DIR, FAULTS, main as _generate_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RESULTS_PATH = os.path.join(ROOT, "results.json")

# ---- palette (validated categorical slots, light mode) ----------------------
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]  # one per channel
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, PAGE = "#e1e0d9", "#fcfcfb", "#f9f9f7"
CRITICAL, SERIOUS = "#d03b3b", "#ec835a"

plt.rcParams.update({
    "figure.facecolor": PAGE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": INK,
    "font.family": "sans-serif", "legend.frameon": False,
})


# ---- data --------------------------------------------------------------------
def load_data():
    """Return (train_df, test_df, labels). Generates the CSVs on first call."""
    train_p = os.path.join(DATA_DIR, "dcs_train.csv")
    test_p = os.path.join(DATA_DIR, "dcs_test.csv")
    if not (os.path.exists(train_p) and os.path.exists(test_p)):
        _generate_data()
    train = pd.read_csv(train_p)
    test = pd.read_csv(test_p)
    labels = test.pop("anomaly").to_numpy()
    return train, test, labels


def standardize(train_df, test_df):
    """z-score both splits with TRAIN statistics only."""
    mu = train_df[CHANNELS].mean()
    sd = train_df[CHANNELS].std()
    return ((train_df[CHANNELS] - mu) / sd).to_numpy(), \
           ((test_df[CHANNELS] - mu) / sd).to_numpy()


def make_windows(X, w, stride=1):
    """(n, d) -> (n_windows, w, d) sliding windows."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(X, (w, X.shape[1])).squeeze(1)[::stride].copy()


def align_scores(win_scores, n, w):
    """Map per-window scores (assigned to the window's LAST step) onto the
    full test timeline; the first w-1 steps get the first score."""
    full = np.empty(n)
    full[:w - 1] = win_scores[0]
    full[w - 1:] = win_scores
    return full


def knn_score(X_train, X_test, k=8):
    """Distance to the k-th nearest healthy training state — a simple,
    model-free anomaly score. Returns (calibration_scores, test_scores),
    the calibration ones computed on the train set itself (self-match
    excluded) so thresholding methods can be tuned on healthy data."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_train)
    calib = nn.kneighbors(X_train)[0][:, -1]
    scores = nn.kneighbors(X_test, n_neighbors=k)[0][:, -1]
    return calib, scores


# ---- evaluation ---------------------------------------------------------------
def _prf(preds, labels):
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def true_segments(labels):
    """[(start, end_exclusive), ...] of contiguous label==1 runs."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], labels, [0]])))
    return list(zip(edges[::2], edges[1::2]))


def point_adjust(preds, labels):
    """SMD/OmniAnomaly protocol: if any point of a true anomalous segment is
    detected, the whole segment counts as detected."""
    adj = preds.copy()
    for s, e in true_segments(labels):
        if adj[s:e].any():
            adj[s:e] = 1
    return adj


def evaluate(scores, labels):
    """Pick the threshold that maximises the RAW point-wise F1, then also
    report the point-adjusted metrics at that same threshold."""
    qs = np.unique(np.quantile(scores, np.linspace(0.70, 0.999, 250)))
    best = None
    for thr in qs:
        preds = (scores >= thr).astype(int)
        p, r, f1 = _prf(preds, labels)
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(thr), "precision": p, "recall": r, "f1": f1}
    preds = (scores >= best["threshold"]).astype(int)
    pa_p, pa_r, pa_f1 = _prf(point_adjust(preds, labels), labels)
    best.update({"pa_precision": pa_p, "pa_recall": pa_r, "pa_f1": pa_f1,
                 "per_fault": fault_coverage(preds)})
    return best


def fault_coverage(preds):
    """Fraction of each injected fault's steps that were flagged."""
    return {name: float(preds[s:e].mean()) for name, s, e in FAULTS}


def save_result(name, metrics):
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    results[name] = {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in metrics.items()}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{name}] F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}  |  point-adjusted F1={metrics['pa_f1']:.3f}")


# ---- plotting -----------------------------------------------------------------
def _shade_truth(ax, labels):
    for s, e in true_segments(labels):
        ax.axvspan(s, e, color=CRITICAL, alpha=0.13, lw=0)


def plot_detection(name, title, test_df, labels, scores, threshold,
                   extra_panel=None, log_score=False):
    """Standard 3-panel result figure:
       (1) the five channels, z-scored and vertically offset, truth shaded
       (2) anomaly score + threshold + detections, truth shaded
       (3) optional method-specific panel drawn by extra_panel(ax)."""
    n_rows = 3 if extra_panel else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.6 * n_rows + 0.8),
                             sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)

    ax = axes[0]
    Xz = (test_df[CHANNELS] - test_df[CHANNELS].mean()) / test_df[CHANNELS].std()
    for i, ch in enumerate(CHANNELS):
        ax.plot(Xz[ch] + 4 * (len(CHANNELS) - 1 - i), color=SERIES[i], lw=0.7)
        ax.text(len(Xz) + 30, 4 * (len(CHANNELS) - 1 - i), ch,
                color=SERIES[i], va="center", fontsize=8)
    _shade_truth(ax, labels)
    ax.set_yticks([])
    ax.set_xlim(0, len(Xz) * 1.14)
    ax.set_title("Test telemetry (z-scored, offset) — true anomalies shaded")

    ax = axes[1]
    ax.plot(scores, color=INK_2, lw=0.8, label="anomaly score")
    ax.axhline(threshold, color=SERIOUS, ls="--", lw=1.2,
               label=f"threshold = {threshold:.3g}")
    det = np.flatnonzero(scores >= threshold)
    ax.plot(det, scores[det], ".", color=CRITICAL, ms=3, label="detected")
    _shade_truth(ax, labels)
    if log_score:
        ax.set_yscale("symlog" if (scores <= 0).any() else "log")
    ax.set_xlim(0, len(scores) * 1.14)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Anomaly score")
    ax.set_xlabel("time step")

    if extra_panel:
        extra_panel(axes[2])

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[{name}] figure -> {os.path.relpath(out, ROOT)}")
