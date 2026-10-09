"""LightGBM supervised anomaly classifier for ATLAS DCS.

*Difficulty: 7 (supervised, gradient boosting).*  Microsoft's gradient-boosting
framework -- like XGBoost but with histogram binning and leaf-wise tree growth,
which makes it markedly faster and lighter on large tabular data while matching
or beating XGBoost's accuracy.  On a 68 TB DCS archive that speed matters.

How it works
------------
Leaf-wise boosting: each new tree grows the leaf with the largest loss
reduction (rather than level-wise), reaching lower loss with fewer nodes.
``is_unbalance=True`` reweights the ~5% positive class.  Score =
``predict_proba[:, 1]``; ``feature_importances_`` (split gain) ranks features.

Why it fits ATLAS DCS
---------------------
Fastest of the boosting family to train, native categorical handling (useful
for ``subsystem``/``element_id``), and a Spark connector (``synapseml`` /
``lightgbm.dask``) for distribution.  Supervised, so pair with an unsupervised
screen for novel faults.

Requires ``lightgbm`` (``pip install lightgbm``).

Run standalone::

    python algorithms/supervised/lightgbm.py
"""

from __future__ import annotations

import os
import sys
import time

# ``lightgbm.py`` would shadow the installed ``lightgbm`` package (its directory
# is on sys.path when run as a script). Drop the script dir, add the root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _THIS_DIR]
sys.path.insert(0, _ROOT)

from utils.pipeline import (finalize_supervised,  # noqa: E402
                            prepare_supervised)

NAME = "lightgbm"


def run(n_estimators: int = 400, num_leaves: int = 31,
        learning_rate: float = 0.05, random_state: int = 42) -> dict:
    """Train a LightGBM classifier and evaluate on the held-out split."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("lightgbm not installed -- run `pip install lightgbm`"
                         ) from exc

    data = prepare_supervised(engineered=True, train_frac=0.6)
    print(f"[{NAME}] train={len(data.train):,} test={len(data.test):,} "
          f"features={len(data.feature_cols)}")

    t0 = time.perf_counter()
    model = LGBMClassifier(
        n_estimators=n_estimators, num_leaves=num_leaves,
        learning_rate=learning_rate, subsample=0.9, colsample_bytree=0.9,
        is_unbalance=True, n_jobs=-1, random_state=random_state, verbose=-1)
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = model.predict_proba(data.X_test)[:, 1]
    predict_time = time.perf_counter() - t1

    return finalize_supervised(NAME, data, scores, train_time, predict_time,
                               feature_importance=model.feature_importances_)


if __name__ == "__main__":
    run()
