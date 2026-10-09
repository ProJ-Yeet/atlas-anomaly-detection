"""
02 - Isolation Forest
=====================
Idea: build random trees that recursively split the feature space at random
values. Anomalous points are "easy to isolate" — they end up in shallow
leaves — so the average path length over many trees becomes an anomaly score.
Trained only on healthy TRAIN states; scored on the test stream.

Like DBSCAN this is a per-timestep (spatial) detector: strong on point and
persistent out-of-range faults, blind to purely temporal anomalies.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from common import utils

NAME = "02_isolation_forest"


def main():
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)

    iforest = IsolationForest(n_estimators=300, max_samples=1024,
                              random_state=42).fit(X_train)
    # score_samples: higher = more normal, so negate
    scores = -iforest.score_samples(X_test)

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    def extra(ax):
        bins = np.linspace(scores.min(), scores.max(), 60)
        ax.hist(scores[labels == 0], bins=bins, color=utils.SERIES[0],
                alpha=0.75, label="normal steps")
        ax.hist(scores[labels == 1], bins=bins, color=utils.CRITICAL,
                alpha=0.75, label="anomalous steps")
        ax.axvline(metrics["threshold"], color=utils.SERIOUS, ls="--", lw=1.2)
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Score distribution — anomalies sit in the short-path tail")
        ax.set_xlabel("isolation score")

    utils.plot_detection(NAME, "Isolation Forest", test_df, labels,
                         scores, metrics["threshold"], extra)


if __name__ == "__main__":
    main()
