"""Isolation Forest -- unsupervised anomaly detection for ATLAS DCS.

*Difficulty: 1 (baseline).*  The recommended first detector for any tabular
anomaly problem: no distance metric to tune, O(n log n) training, microsecond
scoring, and no assumption about the shape of "normal".

How it works
------------
Build many random binary trees.  At each node pick a random feature and a
random split value between that feature's min and max.  A point's **path
length** -- how many splits are needed to isolate it -- is *short* for outliers
(they sit alone in sparse regions, so random cuts separate them quickly) and
*long* for inliers.  The anomaly score is the path length averaged over the
forest and normalised; sklearn returns ``-score_samples`` so that larger ==
more anomalous, which is the convention every script in this project uses.

Why it fits ATLAS DCS
---------------------
* Embarrassingly parallel: trees are independent, so training and scoring shard
  trivially across Spark executors (the friend's distributed repo trains one
  small forest per partition and ensemble-averages -- see the README).
* Cheap enough to run always-on across thousands of channel groups.
* Blind to purely temporal structure (it scores each row independently), so it
  is a *screen*, not the whole answer -- the deep sequence models cover the
  contextual / collective faults it misses.

Run standalone::

    python algorithms/unsupervised/isolation_forest.py
"""

from __future__ import annotations

import os
import sys
import time

# --- make the package importable when run as a plain script --------------- #
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from sklearn.ensemble import IsolationForest  # noqa: E402

from utils.pipeline import (finalize_unsupervised,  # noqa: E402
                            prepare_unsupervised)

NAME = "isolation_forest"


def run(n_estimators: int = 200, contamination: float = 0.05,
        random_state: int = 42) -> dict:
    """Train an Isolation Forest, score every row, evaluate, and save outputs.

    Parameters
    ----------
    n_estimators
        Number of random trees in the forest.
    contamination
        Expected anomaly fraction; only affects sklearn's internal ``offset_``
        -- we threshold on the continuous score ourselves, so this is just a
        sensible prior (~5% here).
    random_state
        Reproducibility seed.
    """
    data = prepare_unsupervised(engineered=True, scale=True)
    print(f"[{NAME}] rows={len(data.df):,}  features={len(data.feature_cols)}")

    t0 = time.perf_counter()
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(1024, len(data.X)),   # subsampling sharpens isolation
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(data.X)
    train_time = time.perf_counter() - t0

    # Anomaly score: negate sklearn's score so larger == more anomalous.
    t1 = time.perf_counter()
    scores = -model.score_samples(data.X)
    predict_time = time.perf_counter() - t1

    return finalize_unsupervised(NAME, data, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
