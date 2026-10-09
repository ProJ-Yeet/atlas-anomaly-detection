"""XGBoost supervised anomaly classifier for ATLAS DCS.

*Difficulty: 7.*  Gradient-boosted decision trees -- the top performer on
tabular anomaly benchmarks (and #1 in the project's own ranking).  Where the
Random Forest builds independent trees and averages, boosting builds trees
**sequentially**, each correcting the residual errors of the ensemble so far,
which usually squeezes out higher precision/recall on structured tabular data.

How it works
------------
``XGBClassifier`` fits an additive model of shallow trees by gradient descent on
the logistic loss.  ``scale_pos_weight`` (= negatives/positives) rebalances the
~5% positive rate.  The anomaly score is the predicted positive-class
probability; feature gains give an interpretable importance ranking.

Why it fits ATLAS DCS
---------------------
Best-in-class accuracy on engineered tabular features (rolling stats, rates,
interactions), handles missing values natively, and distributes on Spark via
``xgboost.spark`` for the eventual big-data deployment.  Same caveat as all
supervised models: it recognises trained fault types, not novel ones.

Requires ``xgboost`` (``pip install xgboost``).

Run standalone::

    python algorithms/supervised/xgboost.py
"""

from __future__ import annotations

import os
import sys
import time

# This file is named ``xgboost.py``; its own directory is on sys.path when run
# as a script, which would shadow the installed ``xgboost`` package. Drop the
# script directory from the path and add the package root instead.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _THIS_DIR]
sys.path.insert(0, _ROOT)

from utils.pipeline import (finalize_supervised,  # noqa: E402
                            prepare_supervised)

NAME = "xgboost"


def run(n_estimators: int = 400, max_depth: int = 6,
        learning_rate: float = 0.1, random_state: int = 42) -> dict:
    """Train an XGBoost classifier and evaluate on the held-out split."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("xgboost not installed -- run `pip install xgboost`"
                         ) from exc

    data = prepare_supervised(engineered=True, train_frac=0.6)
    pos = max(int(data.y_train.sum()), 1)
    neg = int((data.y_train == 0).sum())
    print(f"[{NAME}] train={len(data.train):,} test={len(data.test):,} "
          f"features={len(data.feature_cols)} scale_pos_weight={neg / pos:.1f}")

    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=neg / pos, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=random_state)
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = model.predict_proba(data.X_test)[:, 1]
    predict_time = time.perf_counter() - t1

    return finalize_supervised(NAME, data, scores, train_time, predict_time,
                               feature_importance=model.feature_importances_)


if __name__ == "__main__":
    run()
