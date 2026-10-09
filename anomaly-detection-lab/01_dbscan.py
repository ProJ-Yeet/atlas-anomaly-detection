"""
01 - Density-based clustering: DBSCAN
=====================================
Idea: healthy detector states form dense clusters in sensor space; points in
low-density regions are outliers. DBSCAN labels such points as noise (-1).

Here we run DBSCAN per time bucket (a simplified T-DBSCAN, as in DeepHYDRA):
the test stream is split into buckets and clustered bucket-by-bucket, so a
"dense" state is judged against its local time context. For a continuous
anomaly score (needed for threshold sweeps) we use the k-NN distance to the
healthy TRAIN states — exactly the quantity DBSCAN thresholds with eps.

Expected behaviour on our data: catches point spikes and the cascading fault
(states far from any healthy cluster) but misses the contextual phase-shift
anomaly, whose individual values are perfectly normal. That blind spot is why
the proposal pairs DBSCAN with sequence models.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from common import utils

NAME = "01_dbscan"
K = 8            # min_samples / k for the k-NN distance score
EPS = 1.2        # DBSCAN radius in z-scored units
BUCKET = 500     # time-bucket length for the T-DBSCAN pass


def main():
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)

    # continuous score: distance to the K-th nearest healthy training state
    nn = NearestNeighbors(n_neighbors=K).fit(X_train)
    scores = nn.kneighbors(X_test)[0][:, -1]

    # T-DBSCAN pass (per time bucket) for the cluster picture
    dbscan_noise = np.zeros(len(X_test), dtype=bool)
    for s in range(0, len(X_test), BUCKET):
        seg = X_test[s:s + BUCKET]
        lab = DBSCAN(eps=EPS, min_samples=K).fit_predict(seg)
        dbscan_noise[s:s + BUCKET] = lab == -1

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    pca = PCA(n_components=2).fit(X_train)
    P = pca.transform(X_test)

    def extra(ax):
        normal = ~dbscan_noise
        ax.scatter(P[normal, 0], P[normal, 1], s=3, color=utils.SERIES[0],
                   alpha=0.25, label="in a dense cluster")
        ax.scatter(P[dbscan_noise, 0], P[dbscan_noise, 1], s=8,
                   color=utils.CRITICAL, label="DBSCAN noise (-1)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Test states in PCA plane — T-DBSCAN cluster membership")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    utils.plot_detection(NAME, "DBSCAN (density-based clustering, T-DBSCAN buckets)",
                         test_df, labels, scores, metrics["threshold"], extra)


if __name__ == "__main__":
    main()
