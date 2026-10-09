"""Shared metrics and thresholding for every detector.

All detectors emit a continuous **anomaly score** (higher == more anomalous).
This module turns a score stream + ground truth into the standard report:
accuracy / precision / recall / F1 / ROC-AUC / PR-AUC / confusion counts, plus
the *point-adjusted* F1 (the SMD / OmniAnomaly evaluation protocol) which
answers "did an alarm fire *during* the incident?" rather than "was every step
localised?".

Threshold selection sweeps score quantiles and keeps the threshold that
maximises raw point-wise F1.  This is an optimistic-but-uniform convention:
identical for all models, so the *comparison* is fair; a production system must
instead calibrate the threshold on healthy data (Extreme-Value-Theory / POT).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def best_threshold(scores: np.ndarray, y_true: np.ndarray,
                   n_quantiles: int = 200) -> float:
    """Threshold (on ``scores``) maximising raw point-wise F1.

    Solved exactly in a single ``O(n log n)`` pass instead of re-scoring the
    whole prediction vector once per candidate.  Sorting the scores descending
    makes "flag the top *k*" the only family of thresholds worth considering,
    and a cumulative sum of the sorted labels gives TP(k) for every *k* at once.
    Since ``TP + FP == k`` and ``TP + FN == n_pos``,

        F1(k) = 2 * TP(k) / (k + n_pos)

    so every candidate is scored in two vector operations.  Only the last index
    of a run of equal scores is a realisable cut (``scores >= t`` takes ties as
    a block), and the search is capped at ``k <= n/2`` -- the same "never flag
    more than half the stream" band the previous quantile sweep enforced.

    ``n_quantiles`` is kept for backwards compatibility and ignored: the sweep
    is now exhaustive inside that band, so there is nothing left to discretise.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_true).astype(int)
    n = len(scores)
    n_pos = int(y.sum())
    if n == 0:
        return 0.0
    if n_pos == 0:                       # no positives -- F1 is 0 everywhere
        return float(scores.max())

    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    tp = np.cumsum(y[order])                       # TP when flagging the top k
    k = np.arange(1, n + 1)                        # predicted-positive count

    # A cut is only realisable at the end of a run of equal scores.
    realisable = np.empty(n, dtype=bool)
    realisable[:-1] = s_sorted[:-1] != s_sorted[1:]
    realisable[-1] = True
    realisable &= k <= max(1, n // 2)

    f1 = np.zeros(n)
    f1[realisable] = 2.0 * tp[realisable] / (k[realisable] + n_pos)
    return float(s_sorted[int(np.argmax(f1))])


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Point-adjusted predictions: if any step of a true anomalous segment is
    flagged, the whole contiguous segment is counted as detected.

    Segment boundaries come from the edges of ``y_true`` and "was anything
    flagged inside this segment" from a prefix sum of ``y_pred``, so the whole
    adjustment is vectorised rather than a per-row Python walk.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int).copy()
    n = len(y_true)
    if n == 0:
        return y_pred

    edges = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)            # exclusive
    if len(starts) == 0:
        return y_pred

    prefix = np.concatenate(([0], np.cumsum(y_pred)))
    hit = (prefix[ends] - prefix[starts]) > 0     # segment contains a flag
    if not hit.any():
        return y_pred

    # Paint every hit segment via a +1/-1 difference array + cumulative sum.
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, starts[hit], 1)
    np.add.at(delta, ends[hit], -1)
    y_pred[np.cumsum(delta)[:n] > 0] = 1
    return y_pred


def compute_metrics(y_true: np.ndarray, scores: np.ndarray,
                    threshold: Optional[float] = None) -> Dict[str, float]:
    """Full metric dict for one detector.

    If ``threshold`` is None it is chosen by :func:`best_threshold`.  Scores are
    used directly for ROC-AUC / PR-AUC (threshold-free); the binary metrics use
    the thresholded predictions.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if threshold is None:
        threshold = best_threshold(scores, y_true)
    pred = (scores >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    # Threshold-free ranking metrics (guard against single-class y).
    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = float("nan")

    pa_pred = point_adjust(y_true, pred)
    pa_p, pa_r, pa_f1, _ = precision_recall_fscore_support(
        y_true, pa_pred, average="binary", zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "pa_f1": float(pa_f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def print_report(name: str, m: Dict[str, float]) -> None:
    """Pretty-print a metric dict to stdout."""
    print(f"\n=== {name} ===")
    print(f"  threshold : {m['threshold']:.4f}")
    print(f"  accuracy  : {m['accuracy']:.4f}")
    print(f"  precision : {m['precision']:.4f}   recall: {m['recall']:.4f}"
          f"   F1: {m['f1']:.4f}")
    print(f"  ROC-AUC   : {m['roc_auc']:.4f}   PR-AUC: {m['pr_auc']:.4f}"
          f"   PA-F1: {m['pa_f1']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")


def save_predictions(out_dir: str, name: str, meta: pd.DataFrame,
                     scores: np.ndarray, y_pred: np.ndarray,
                     metrics: Optional[Dict[str, float]] = None) -> str:
    """Write ``<name>_predictions.csv`` (+ ``<name>_metrics.json``).

    ``meta`` supplies the identifying columns (row_id / element_id / timestamp /
    subsystem) that are available; whatever subset exists is carried through.
    """
    os.makedirs(out_dir, exist_ok=True)
    keep = [c for c in ["row_id", "timestamp", "element_id", "subsystem",
                        "anomaly"] if c in meta.columns]
    out = meta[keep].copy()
    out["anomaly_score"] = np.asarray(scores, dtype=float)
    out["predicted"] = np.asarray(y_pred, dtype=int)
    pred_path = os.path.join(out_dir, f"{name}_predictions.csv")
    out.to_csv(pred_path, index=False)
    if metrics is not None:
        with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
    return pred_path
