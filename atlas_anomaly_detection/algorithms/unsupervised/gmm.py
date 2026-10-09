"""Gaussian Mixture Model anomaly detection for ATLAS DCS.

*Difficulty: 5.*  The probabilistic cousin of K-Means: model the healthy data as
a mixture of ``k`` Gaussians and score each row by its **negative log-likelihood**
under that mixture.  Unlike K-Means it captures elliptical, correlated clusters
(full covariance) and yields a calibrated probability rather than a raw distance.

How it works
------------
Fit ``GaussianMixture`` (EM) on the feature matrix; ``score_samples`` returns the
log-likelihood, so the anomaly score is its negation -- low-density rows (far in
the tails of every component) score high.  A subsample can be used to fit if the
covariance estimation is slow, but 100k x ~40 with a handful of components is
fine on CPU.

Why it fits ATLAS DCS
---------------------
Native to Spark MLlib, gives an interpretable likelihood ("this state has
probability 1e-7 under normal operation"), and full covariance models the
channel correlations that K-Means ignores.  Still per-timestep and assumes
Gaussian components, so heavy-tailed or highly nonlinear structure is
approximated coarsely.

Run standalone::

    python algorithms/unsupervised/gmm.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from sklearn.mixture import GaussianMixture  # noqa: E402

from utils.pipeline import (finalize_unsupervised,  # noqa: E402
                            prepare_unsupervised)

NAME = "gmm"


def run(k: int = 6, covariance_type: str = "diag",
        random_state: int = 42) -> dict:
    """Fit a Gaussian mixture, score each row by negative log-likelihood.

    Parameters
    ----------
    k
        Number of mixture components.
    covariance_type
        ``"diag"`` is fast and robust in ~40-D; ``"full"`` models all channel
        correlations but is heavier and can be ill-conditioned.
    random_state
        Reproducibility seed.
    """
    data = prepare_unsupervised(engineered=True, scale=True)
    print(f"[{NAME}] rows={len(data.df):,}  features={len(data.feature_cols)}")

    t0 = time.perf_counter()
    model = GaussianMixture(n_components=k, covariance_type=covariance_type,
                            reg_covar=1e-4, random_state=random_state,
                            max_iter=200)
    model.fit(data.X)
    train_time = time.perf_counter() - t0
    print(f"  converged={model.converged_}  n_iter={model.n_iter_}")

    # Negative log-likelihood: larger == less probable == more anomalous.
    t1 = time.perf_counter()
    scores = -model.score_samples(data.X)
    predict_time = time.perf_counter() - t1

    return finalize_unsupervised(NAME, data, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
