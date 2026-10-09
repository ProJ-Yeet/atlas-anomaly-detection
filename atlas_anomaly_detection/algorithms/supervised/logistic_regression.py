"""Logistic Regression supervised anomaly classifier for ATLAS DCS.

*Difficulty: 6 (supervised, linear baseline).*  The simplest supervised model --
a linear decision boundary in engineered-feature space, fit by maximising the
class-weighted logistic likelihood.  Its job here is to be the *interpretable
floor*: any complex model must beat it to justify its cost.

How it works
------------
Learns a weight per feature; the anomaly score is the predicted positive-class
probability (sigmoid of the linear combination).  ``class_weight="balanced"``
offsets the ~5% positive rate.  The absolute standardized coefficients give a
transparent "which sensor drives the alarm" ranking -- often exactly what an
operator wants.

Why it fits ATLAS DCS
---------------------
Near-zero training/scoring cost, fully interpretable coefficients (property g),
and trivially distributed (Spark MLlib ``LogisticRegression``).  It can only
draw a linear boundary, so it underperforms the tree ensembles on nonlinear
fault signatures -- but it is the right first supervised sanity check.

Run standalone::

    python algorithms/supervised/logistic_regression.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from utils.pipeline import (finalize_supervised,  # noqa: E402
                            prepare_supervised)

NAME = "logistic_regression"


def run(C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> dict:
    """Train a logistic-regression classifier and evaluate on the test split.

    Parameters
    ----------
    C
        Inverse L2 regularisation strength.
    max_iter
        Maximum solver iterations.
    random_state
        Reproducibility seed.
    """
    data = prepare_supervised(engineered=True, train_frac=0.6)
    print(f"[{NAME}] train={len(data.train):,} test={len(data.test):,} "
          f"features={len(data.feature_cols)}")

    t0 = time.perf_counter()
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter,
                               random_state=random_state)
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = model.predict_proba(data.X_test)[:, 1]
    predict_time = time.perf_counter() - t1

    # |coefficient| as feature importance (features are standardized).
    importance = np.abs(model.coef_.ravel())
    return finalize_supervised(NAME, data, scores, train_time, predict_time,
                               feature_importance=importance)


if __name__ == "__main__":
    run()
