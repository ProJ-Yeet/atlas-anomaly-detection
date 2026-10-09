"""PCA reconstruction-error anomaly detection for ATLAS DCS.

*Difficulty: 3.*  The simplest "reconstruction" detector and the linear
ancestor of every autoencoder in this project.

How it works
------------
Principal Component Analysis finds the orthogonal directions of greatest
variance.  Healthy, tightly-correlated DCS telemetry lives on a low-dimensional
linear subspace (cooling -> temperature -> current is almost a line).  Keep the
top ``n_components`` and reconstruct each row from them; a healthy row is
reconstructed almost perfectly, while an anomaly -- which by definition leaves
the normal correlation structure -- lands off the subspace and has large
**reconstruction error**.  Score = squared reconstruction error.

Why it fits ATLAS DCS
---------------------
Trivial to train, closed-form, interpretable (you can read off *which* channels
drove the residual), and a natural fit for Spark MLlib's distributed PCA.  It is
linear, so it cannot model nonlinear manifolds or temporal shape -- but it is an
excellent, near-free baseline and a sanity check for the deep models.

Run standalone::

    python algorithms/unsupervised/pca.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from utils.pipeline import (finalize_unsupervised,  # noqa: E402
                            prepare_unsupervised)

NAME = "pca"


def run(n_components: int = 10, random_state: int = 42) -> dict:
    """Fit PCA, reconstruct every row, score by squared reconstruction error.

    Parameters
    ----------
    n_components
        Number of principal components retained (the "normal" subspace).
    random_state
        Reproducibility seed for the randomized SVD solver.
    """
    data = prepare_unsupervised(engineered=True, scale=True)
    print(f"[{NAME}] rows={len(data.df):,}  features={len(data.feature_cols)}")

    n_components = min(n_components, data.X.shape[1])

    t0 = time.perf_counter()
    model = PCA(n_components=n_components, random_state=random_state)
    model.fit(data.X)
    train_time = time.perf_counter() - t0
    print(f"  explained variance (top {n_components}): "
          f"{model.explained_variance_ratio_.sum():.3f}")

    # Reconstruct and score by per-row squared error.
    t1 = time.perf_counter()
    reconstructed = model.inverse_transform(model.transform(data.X))
    scores = np.mean((data.X - reconstructed) ** 2, axis=1)
    predict_time = time.perf_counter() - t1

    return finalize_unsupervised(NAME, data, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
