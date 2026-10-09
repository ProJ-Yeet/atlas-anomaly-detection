"""DBSCAN / density-based anomaly detection for ATLAS DCS.

*Difficulty: 2.*  Density clustering with no assumed cluster shape and no
"number of clusters" parameter: dense regions are "normal", points reachable
from no dense core are anomalies (label -1).

Turning DBSCAN into a *continuous* score
----------------------------------------
Plain DBSCAN outputs a hard -1/cluster label, but the threshold sweep needs a
continuous score.  We use the quantity DBSCAN itself compares against ``eps``:
the **distance to the k-th nearest neighbour** in a healthy reference sample.
``eps`` is estimated as a high percentile of those k-NN distances -- exactly the
heuristic used in the friend's distributed repo (95th percentile of k-NN
distances from a driver sample).  A genuine ``sklearn`` DBSCAN is also run on a
subsample to report the cluster/noise structure.

Why it fits ATLAS DCS
---------------------
Cheap, explainable ("this state is far from any state seen recently"), and
shardable per time bucket / channel group.  Like Isolation Forest it is
per-timestep, so it is blind to temporal shape -- a fast first-pass screen.

Run standalone::

    python algorithms/unsupervised/dbscan.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from utils.pipeline import (finalize_unsupervised,  # noqa: E402
                            prepare_unsupervised)

NAME = "dbscan"


def run(k: int = 10, n_reference: int = 8000, eps_percentile: float = 95.0,
        random_state: int = 42) -> dict:
    """Score every row by k-NN distance to a healthy reference sample.

    Parameters
    ----------
    k
        Neighbour rank used for the density score (distance to the k-th
        nearest reference point).
    n_reference
        Size of the random reference subsample the k-NN index is built on
        (keeps scoring O(n log n) instead of DBSCAN's O(n^2) on 100k rows).
    eps_percentile
        Percentile of reference k-NN distances used as the DBSCAN ``eps`` for
        the illustrative subsample clustering.
    random_state
        Reproducibility seed.
    """
    data = prepare_unsupervised(engineered=True, scale=True)
    print(f"[{NAME}] rows={len(data.df):,}  features={len(data.feature_cols)}")

    rng = np.random.default_rng(random_state)
    ref_idx = rng.choice(len(data.X), size=min(n_reference, len(data.X)),
                         replace=False)
    reference = data.X[ref_idx]

    # --- fit the k-NN density index on the reference ---------------------- #
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(reference)
    # eps from the reference's own k-NN distances (self-distances included).
    ref_dist, _ = nn.kneighbors(reference)
    eps = float(np.percentile(ref_dist[:, -1], eps_percentile))
    train_time = time.perf_counter() - t0

    # --- score all rows by distance to k-th reference neighbour ----------- #
    t1 = time.perf_counter()
    dist, _ = nn.kneighbors(data.X)
    scores = dist[:, -1]
    predict_time = time.perf_counter() - t1

    # --- illustrative DBSCAN clustering on a subsample -------------------- #
    sub = data.X[rng.choice(len(data.X), size=min(5000, len(data.X)),
                            replace=False)]
    labels = DBSCAN(eps=eps, min_samples=k).fit_predict(sub)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  eps={eps:.3f}  subsample DBSCAN: {n_clusters} clusters, "
          f"{np.mean(labels == -1):.1%} noise")

    return finalize_unsupervised(NAME, data, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
