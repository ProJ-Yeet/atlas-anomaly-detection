"""Support Vector Machine supervised anomaly classifier for ATLAS DCS.

*Difficulty: 6-7 (supervised, kernel method).*  An RBF-kernel SVM finds the
maximum-margin boundary between healthy and faulty states in an implicit
high-dimensional feature space, capturing nonlinear signatures a linear model
cannot.

Scalability note
----------------
Kernel SVM training is roughly O(n^2)-O(n^3), which is infeasible on 60k training
rows.  We therefore fit on a **stratified subsample** (``train_subsample`` rows,
keeping all positives) -- standard practice for kernel SVMs on large data.  The
anomaly score is the signed ``decision_function`` (distance to the margin;
higher == more anomalous).  For a genuinely large-scale linear alternative, swap
in ``SGDClassifier(loss="hinge")`` (noted in the README).

Why it fits ATLAS DCS
---------------------
Strong on nonlinear, moderate-dimensional problems and margin-based scoring is
naturally calibrated for thresholding.  Poor scaling and no native feature
importance are the trade-offs; it is included for methodological completeness
and as a nonlinear contrast to logistic regression.

Run standalone::

    python algorithms/supervised/svm.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from joblib import Parallel, delayed, effective_n_jobs  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from utils.pipeline import (finalize_supervised,  # noqa: E402
                            prepare_supervised)

NAME = "svm"


def _parallel_decision_function(model: SVC, X: np.ndarray,
                                n_jobs: int = -1) -> np.ndarray:
    """``model.decision_function(X)`` fanned out over row blocks.

    libsvm scores rows one at a time on a single thread, which makes prediction
    -- not training -- the SVM's dominant cost here (40k test rows against ~2.7k
    support vectors).  Each row's kernel evaluation is independent, so splitting
    the matrix and concatenating the parts is *exact*, not an approximation; it
    just uses the other cores.

    The ``threading`` backend is deliberate: libsvm releases the GIL inside
    ``decision_function``, so threads get real parallelism with no transfer
    cost, whereas the default process backend has to spawn interpreters and
    pickle the model per block -- measured *slower than serial* on Windows
    (18.6s across 8 processes vs 12.8s serial vs 5.4s across 8 threads).
    """
    n_blocks = max(1, effective_n_jobs(n_jobs))
    if n_blocks == 1 or len(X) < 8192:
        return model.decision_function(X)
    blocks = np.array_split(X, n_blocks)
    parts = Parallel(n_jobs=n_blocks, backend="threading")(
        delayed(model.decision_function)(b) for b in blocks)
    return np.concatenate(parts)


def _stratified_subsample(X: np.ndarray, y: np.ndarray, n: int,
                          seed: int) -> tuple:
    """Keep every positive plus a random negative sample, up to ``n`` rows."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_neg = min(len(neg), max(n - len(pos), n // 2))
    keep = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep]


def run(train_subsample: int = 8000, C: float = 2.0, gamma: str = "scale",
        random_state: int = 42) -> dict:
    """Train an RBF-kernel SVM on a subsample; score the full test split."""
    data = prepare_supervised(engineered=True, train_frac=0.6)
    Xs, ys = _stratified_subsample(data.X_train, data.y_train,
                                   train_subsample, random_state)
    print(f"[{NAME}] train(subsample)={len(Xs):,} test={len(data.test):,} "
          f"features={len(data.feature_cols)} pos={int(ys.sum())}")

    t0 = time.perf_counter()
    # cache_size is the kernel-cache budget in MB; the 200 MB default forces
    # libsvm to recompute kernel rows it has already evaluated.
    model = SVC(C=C, kernel="rbf", gamma=gamma, class_weight="balanced",
                cache_size=1000, random_state=random_state)
    model.fit(Xs, ys)
    train_time = time.perf_counter() - t0

    # Signed distance to the separating hyperplane as the anomaly score.
    t1 = time.perf_counter()
    scores = _parallel_decision_function(model, data.X_test)
    predict_time = time.perf_counter() - t1

    return finalize_supervised(NAME, data, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
