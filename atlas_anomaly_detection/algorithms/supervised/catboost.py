"""CatBoost supervised anomaly classifier for ATLAS DCS.

*Difficulty: 7 (supervised, gradient boosting).*  Yandex's gradient-boosting
library, distinguished by **ordered boosting** (a permutation-driven scheme that
reduces the prediction-shift bias of standard boosting) and first-class
categorical support.  Robust out-of-the-box with little tuning.

How it works
------------
Builds symmetric (oblivious) decision trees with ordered target statistics for
categoricals.  ``auto_class_weights="Balanced"`` handles the ~5% positive rate.
Score = ``predict_proba[:, 1]``; ``feature_importances_`` (PredictionValuesChange)
ranks the drivers.

Why it fits ATLAS DCS
---------------------
Excellent default accuracy, native handling of high-cardinality categoricals
like ``element_id`` (thousands of DCS channels), and strong resistance to
overfitting -- valuable when confirmed-fault labels are scarce.  GPU-capable and
distributable.  Supervised, so combine with an unsupervised screen.

Requires ``catboost`` (``pip install catboost``).

Run standalone::

    python algorithms/supervised/catboost.py
"""

from __future__ import annotations

import os
import sys
import time

# ``catboost.py`` would shadow the installed ``catboost`` package. Drop the
# script directory from sys.path and add the package root instead.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _THIS_DIR]
sys.path.insert(0, _ROOT)

from utils.pipeline import (finalize_supervised,  # noqa: E402
                            prepare_supervised)

NAME = "catboost"


def run(iterations: int = 400, depth: int = 6, learning_rate: float = 0.05,
        random_state: int = 42) -> dict:
    """Train a CatBoost classifier and evaluate on the held-out split."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("catboost not installed -- run `pip install catboost`"
                         ) from exc

    data = prepare_supervised(engineered=True, train_frac=0.6)
    print(f"[{NAME}] train={len(data.train):,} test={len(data.test):,} "
          f"features={len(data.feature_cols)}")

    t0 = time.perf_counter()
    model = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        auto_class_weights="Balanced", random_seed=random_state,
        verbose=0, allow_writing_files=False)
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = model.predict_proba(data.X_test)[:, 1]
    predict_time = time.perf_counter() - t1

    return finalize_supervised(NAME, data, scores, train_time, predict_time,
                               feature_importance=model.feature_importances_)


if __name__ == "__main__":
    run()
