"""High-level data-prep helpers so each algorithm script stays short.

These bundle the standard steps -- load -> feature-engineer -> (split) -> matrix
-> scale -> attach labels-for-eval -- into two calls:

* :func:`prepare_unsupervised` for the unsupervised detectors (labels are
  attached for *scoring only*; the returned feature matrix never contains them).
* :func:`prepare_supervised` for the supervised classifiers (temporal
  train/test split, labels returned separately).

Every algorithm importing these gets an identical data contract, which is what
makes the cross-model comparison meaningful.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def bootstrap_path() -> str:
    """Insert the package root on ``sys.path`` so ``python algorithms/.../x.py``
    can ``import utils`` / ``import evaluation``.  Returns the package root.

    Call this at the very top of a standalone script, *before* importing the
    project's own modules.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


@dataclass
class UnsupervisedData:
    """Container for everything an unsupervised script needs."""

    df: pd.DataFrame            # meta + features + 'anomaly' (eval only)
    feature_cols: List[str]
    X: np.ndarray               # scaled feature matrix (n_rows, n_feat)
    y: np.ndarray               # ground truth (eval only)


@dataclass
class SupervisedData:
    """Container for everything a supervised script needs."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


#: Prepared-data caches.  Every one of the 18 detectors asks for the *same*
#: bundle, and building it (CSV parse -> feature engineering -> matrix -> scale)
#: costs about a second each time.  The bundles are treated as read-only by all
#: callers -- the deep models copy the frame before touching it -- so they are
#: memoised per argument set and shared across a run.
_UNSUP_CACHE: dict = {}
_SUP_CACHE: dict = {}


def prepare_unsupervised(engineered: bool = True, scale: bool = True
                         ) -> UnsupervisedData:
    """Load the unlabeled dataset, engineer features, scale, attach labels.

    Memoised per ``(engineered, scale)``; treat the returned bundle as
    read-only.
    """
    from utils.preprocessing import (attach_ground_truth, feature_columns,
                                      get_matrix, load_unlabeled, standardize)
    from utils.feature_engineering import add_features

    key = (engineered, scale)
    if key in _UNSUP_CACHE:
        return _UNSUP_CACHE[key]

    df = load_unlabeled()
    if engineered:
        df = add_features(df)
    df = attach_ground_truth(df)              # 'anomaly' added -- eval only
    feats = feature_columns(df)               # excludes labels + meta
    X = get_matrix(df, feats)
    if scale:
        X, _, _ = standardize(X)
    y = df["anomaly"].to_numpy().astype(int)
    bundle = UnsupervisedData(df=df, feature_cols=feats, X=X, y=y)
    _UNSUP_CACHE[key] = bundle
    return bundle


_REF_CACHE = None


def reference_element_and_rows():
    """The single element (and its test-window row_ids) that every model's
    time-series panel displays, so those panels are directly comparable.

    Chosen deterministically as the element whose held-out **test window** (last
    40% of its timeline) contains the most anomalous rows -- guaranteeing a
    visible, fault-rich example that is identical across all 18 figures and
    identical for supervised (test-only) and unsupervised (all-rows) models.
    """
    global _REF_CACHE
    if _REF_CACHE is None:
        from utils.preprocessing import load_labeled, temporal_split
        df = load_labeled()
        _, test = temporal_split(df, train_frac=0.6)
        counts = test.groupby("element_id")["anomaly"].sum()
        el = str(counts.idxmax())
        rows = set(test[test["element_id"] == el]["row_id"].tolist())
        _REF_CACHE = (el, rows)
    return _REF_CACHE


def finalize_unsupervised(name: str, data: "UnsupervisedData",
                          scores: np.ndarray, train_time: float,
                          predict_time: float,
                          channel: str = "temperature") -> dict:
    """Common tail for unsupervised scripts: metrics, save, 9-panel figure.

    Picks the element with the highest mean score for the time-series panels so
    a real incident is visible, evaluates against the eval-only labels, writes
    ``figures/<name>_predictions.csv`` + ``<name>_metrics.json`` + the figure.
    """
    import os
    from evaluation.metrics import (compute_metrics, print_report,
                                    save_predictions)
    from utils.visualization import full_report_figure

    fig_dir = os.path.join(bootstrap_path(), "figures")
    metrics = compute_metrics(data.y, scores)
    metrics["train_time_s"] = float(train_time)
    metrics["predict_time_s"] = float(predict_time)
    metrics["n_scored"] = int(len(scores))
    metrics["family"] = "unsupervised"
    print_report(name, metrics)
    print(f"  train_time={train_time:.2f}s  predict_time={predict_time:.2f}s")

    y_pred = (scores >= metrics["threshold"]).astype(int)
    save_predictions(fig_dir, name, data.df, scores, y_pred, metrics)

    # Time-series panels use the SAME reference element+window for every model
    # (metric panels below use the full scored dataset, not a sample).
    ref_el, ref_rows = reference_element_and_rows()
    mask = data.df["row_id"].isin(ref_rows).to_numpy()
    if not mask.any():                      # fallback: whole element
        mask = (data.df["element_id"] == ref_el).to_numpy()
    full_report_figure(
        name, scores=scores, threshold=metrics["threshold"], y_true=data.y,
        X_for_pca=data.X, ts_channel=data.df.loc[mask, channel].to_numpy(),
        ts_scores=scores[mask], ts_y=data.y[mask],
        channel_name=f"{channel} - {ref_el} (display channel; model uses all "
                     f"features)")
    print(f"  figure -> figures/{name}.png")
    return metrics


def finalize_supervised(name: str, data: "SupervisedData", scores: np.ndarray,
                        train_time: float, predict_time: float,
                        feature_importance=None,
                        channel: str = "temperature") -> dict:
    """Common tail for supervised scripts: metrics on the held-out test split,
    save, and the 9-panel figure (with a feature-importance panel)."""
    import os
    from evaluation.metrics import (compute_metrics, print_report,
                                    save_predictions)
    from utils.visualization import full_report_figure

    fig_dir = os.path.join(bootstrap_path(), "figures")
    metrics = compute_metrics(data.y_test, scores)
    metrics["train_time_s"] = float(train_time)
    metrics["predict_time_s"] = float(predict_time)
    metrics["n_scored"] = int(len(scores))
    metrics["family"] = "supervised"
    print_report(name, metrics)
    print(f"  train_time={train_time:.2f}s  predict_time={predict_time:.2f}s")

    y_pred = (scores >= metrics["threshold"]).astype(int)
    save_predictions(fig_dir, name, data.test, scores, y_pred, metrics)

    # Same reference element+window as every other model (see above).
    ref_el, ref_rows = reference_element_and_rows()
    mask = data.test["row_id"].isin(ref_rows).to_numpy()
    if not mask.any():
        mask = (data.test["element_id"] == ref_el).to_numpy()
    full_report_figure(
        name, scores=scores, threshold=metrics["threshold"],
        y_true=data.y_test, X_for_pca=data.X_test,
        ts_channel=data.test.loc[mask, channel].to_numpy(),
        ts_scores=scores[mask], ts_y=data.y_test[mask],
        feature_names=data.feature_cols, feature_importance=feature_importance,
        channel_name=f"{channel} - {ref_el} (display channel; model uses all "
                     f"features)")
    print(f"  figure -> figures/{name}.png")
    return metrics


def prepare_supervised(engineered: bool = True, train_frac: float = 0.6,
                       scale: bool = True) -> SupervisedData:
    """Load the labeled dataset, engineer features, temporal split, scale.

    Memoised per ``(engineered, train_frac, scale)``; treat the returned bundle
    as read-only.
    """
    from utils.preprocessing import (feature_columns, get_matrix, load_labeled,
                                      standardize, temporal_split)
    from utils.feature_engineering import add_features

    key = (engineered, train_frac, scale)
    if key in _SUP_CACHE:
        return _SUP_CACHE[key]

    df = load_labeled()
    if engineered:
        df = add_features(df)
    train, test = temporal_split(df, train_frac)
    feats = feature_columns(train)
    X_train = get_matrix(train, feats)
    X_test = get_matrix(test, feats)
    if scale:
        X_train, X_test, _ = standardize(X_train, X_test)
    y_train = train["anomaly"].to_numpy().astype(int)
    y_test = test["anomaly"].to_numpy().astype(int)
    bundle = SupervisedData(train=train, test=test, feature_cols=feats,
                            X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)
    _SUP_CACHE[key] = bundle
    return bundle
