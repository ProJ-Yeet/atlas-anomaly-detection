"""K-Means distance-to-centroid anomaly detection for ATLAS DCS.

*Difficulty: 4.*  Partition the data into ``k`` clusters (the healthy operating
regimes -- e.g. per-subsystem baselines, day/night phases) and score each row by
its **distance to the nearest centroid**.  A row far from every centroid does
not belong to any known regime and is flagged.

How it works
------------
``MiniBatchKMeans`` scales to 100k rows cheaply.  After fitting, the score is
the Euclidean distance to the assigned centroid; the friend's distributed repo
uses the same idea (flag points beyond ~0.7 sigma of the per-cluster distance).

Why it fits ATLAS DCS
---------------------
Native to Spark MLlib and trivially distributed; interpretable ("nearest normal
regime is 5.3 units away").  Assumes convex, roughly spherical clusters, so it
is weaker than DBSCAN on curved manifolds and, like all spatial methods, blind
to temporal structure.

Run standalone::

    python algorithms/unsupervised/kmeans.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402

from utils.pipeline import (finalize_unsupervised,  # noqa: E402
                            prepare_unsupervised)

NAME = "kmeans"


def run(k: int = 8, random_state: int = 42) -> dict:
    """Fit K-Means, score each row by distance to its nearest centroid.

    Parameters
    ----------
    k
        Number of clusters (healthy operating regimes).
    random_state
        Reproducibility seed.
    """
    data = prepare_unsupervised(engineered=True, scale=True)
    print(f"[{NAME}] rows={len(data.df):,}  features={len(data.feature_cols)}")

    t0 = time.perf_counter()
    model = MiniBatchKMeans(n_clusters=k, random_state=random_state,
                            n_init=10, batch_size=2048)
    assign = model.fit_predict(data.X)
    train_time = time.perf_counter() - t0

    # Distance from each row to its assigned centroid.
    t1 = time.perf_counter()
    centroids = model.cluster_centers_[assign]
    scores = np.linalg.norm(data.X - centroids, axis=1)
    predict_time = time.perf_counter() - t1
    print(f"  cluster sizes: {np.bincount(assign).tolist()}")

    return finalize_unsupervised(NAME, data, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
