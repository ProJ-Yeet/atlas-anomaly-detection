"""Random Forest supervised anomaly classifier for ATLAS DCS.

*Difficulty: 6 (first supervised model).*  Uses the **labeled** dataset: given
engineered features and the ground-truth ``anomaly`` column, learn a direct
classifier.  This is the supervised counterpart to Isolation Forest and the
natural first supervised baseline.

How it works
------------
An ensemble of decision trees, each trained on a bootstrap sample with a random
feature subset at every split; the class probability is the fraction of trees
voting "anomaly".  ``class_weight="balanced"`` compensates for the ~5% positive
rate.  The anomaly score is ``predict_proba[:, 1]``.

Why it fits ATLAS DCS
---------------------
When labels exist (from operator-confirmed incidents or synthetic fault
injection), a supervised model is far more precise than any unsupervised one --
it learns the *specific* fault signatures.  Random Forests are robust to feature
scaling, expose feature importances (interpretability, property g), and are
embarrassingly parallel across trees for Spark.  Caveat: they only recognise
fault *types seen in training* -- pair them with an unsupervised screen to catch
novel anomalies.

Evaluation uses a **temporal** train/test split (first 60% of each element's
timeline trains, last 40% tests) -- never a random shuffle, which would leak the
future into the past.

Run standalone::

    python algorithms/supervised/random_forest.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from utils.pipeline import (finalize_supervised,  # noqa: E402
                            prepare_supervised)

NAME = "random_forest"


def run(n_estimators: int = 200, max_depth: int = 16,
        random_state: int = 42) -> dict:
    """Train a Random Forest classifier and evaluate on the held-out split.

    Parameters
    ----------
    n_estimators
        Number of trees.
    max_depth
        Maximum tree depth (caps overfitting on the imbalanced positives).
    random_state
        Reproducibility seed.
    """
    data = prepare_supervised(engineered=True, train_frac=0.6)
    print(f"[{NAME}] train={len(data.train):,} test={len(data.test):,} "
          f"features={len(data.feature_cols)} "
          f"pos_rate={data.y_train.mean():.2%}")

    t0 = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", n_jobs=-1, random_state=random_state)
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    # Anomaly score = probability of the positive (anomaly) class.
    t1 = time.perf_counter()
    scores = model.predict_proba(data.X_test)[:, 1]
    predict_time = time.perf_counter() - t1

    return finalize_supervised(NAME, data, scores, train_time, predict_time,
                               feature_importance=model.feature_importances_)


if __name__ == "__main__":
    run()
