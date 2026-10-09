"""
03 - Clustering + Deep Embeddings
=================================
Idea: instead of clustering raw sensor values, first learn a compressed latent
representation with an autoencoder over sliding WINDOWS, then do the density
reasoning in that latent space. Because each embedding summarises a whole
window, temporal shape information (which raw per-point clustering throws
away) survives into the clustering stage.

Pipeline: window (32 x 5) -> dense autoencoder -> 8-d latent -> k-NN distance
to healthy TRAIN embeddings = anomaly score. DBSCAN on the latents provides
the cluster picture. This is the "Deep Embedding" bullet of the clustering
family and the conceptual bridge to DeepHYDRA's hybrid design.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from common import utils
from common.torch_utils import DEVICE, batched_apply, fit, loader_from, set_seed

NAME = "03_deep_embedding"
W, LATENT = 32, 8


class AE(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(),
                                 nn.Linear(64, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 64), nn.ReLU(),
                                 nn.Linear(64, d_in))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def main():
    set_seed()
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    Wtr = utils.make_windows(X_train, W).reshape(-1, W * X_train.shape[1])
    Wte = utils.make_windows(X_test, W).reshape(-1, W * X_test.shape[1])

    model = AE(Wtr.shape[1]).to(DEVICE)
    fit(model, loader_from(Wtr),
        lambda m, xb: nn.functional.mse_loss(m(xb)[0], xb),
        epochs=15, tag=NAME)

    Ztr = batched_apply(lambda xb: model(xb)[1], Wtr)
    Zte = batched_apply(lambda xb: model(xb)[1], Wte)

    nn_index = NearestNeighbors(n_neighbors=8).fit(Ztr)
    win_scores = nn_index.kneighbors(Zte)[0][:, -1]
    scores = utils.align_scores(win_scores, len(X_test), W)

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    # cluster picture in the latent plane
    eps = float(np.quantile(nn_index.kneighbors(Ztr)[0][:, -1], 0.98)) * 1.5
    noise = DBSCAN(eps=eps, min_samples=8).fit_predict(Zte) == -1
    P = PCA(n_components=2).fit(Ztr).transform(Zte)
    win_labels = labels[W - 1:]

    def extra(ax):
        ax.scatter(P[~noise, 0], P[~noise, 1], s=3, color=utils.SERIES[0],
                   alpha=0.25, label="dense (healthy shape)")
        ax.scatter(P[noise, 0], P[noise, 1], s=8, color=utils.CRITICAL,
                   label="DBSCAN noise in latent space")
        miss = noise & (win_labels == 0)
        if miss.any():
            ax.scatter(P[miss, 0], P[miss, 1], s=8, color=utils.SERIOUS,
                       label="flagged but truly normal")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Window embeddings (PCA of 8-d latent) — clustering after compression")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    utils.plot_detection(NAME, "Deep Embedding + Clustering (AE latent + kNN/DBSCAN)",
                         test_df, labels, scores, metrics["threshold"], extra)


if __name__ == "__main__":
    main()
