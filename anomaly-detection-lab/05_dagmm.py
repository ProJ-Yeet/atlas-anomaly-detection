"""
05 - DAGMM (Deep Autoencoding Gaussian Mixture Model), simplified
=================================================================
Idea: DAGMM couples two networks trained JOINTLY:
  * compression net: an autoencoder producing a low-d latent z_c plus two
    reconstruction-quality features (relative euclidean error, cosine sim);
  * estimation net: maps z = [z_c, rec-features] to soft mixture memberships
    of a Gaussian Mixture Model fitted on the fly from batch statistics.
The anomaly score is the GMM sample ENERGY -log p(z): points that are hard to
compress OR land in low-density latent regions get high energy. Unlike a
plain AE + separate GMM, the joint training shapes the latent space so that
density estimation works well in it.

(Simplified: dense nets, diagonal-friendly covariance regularisation.)
"""

import numpy as np
import torch
import torch.nn as nn

from common import utils
from common.torch_utils import DEVICE, batched_apply, fit, loader_from, set_seed

NAME = "05_dagmm"
W, ZC, K = 32, 2, 4          # window, compressed latent dim, mixture comps
LAMBDA_E, LAMBDA_C = 0.1, 0.005


class DAGMM(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 64), nn.Tanh(), nn.Linear(64, ZC))
        self.dec = nn.Sequential(nn.Linear(ZC, 64), nn.Tanh(), nn.Linear(64, d_in))
        self.est = nn.Sequential(nn.Linear(ZC + 2, 16), nn.Tanh(),
                                 nn.Dropout(0.3), nn.Linear(16, K), nn.Softmax(-1))
        self.register_buffer("phi", torch.ones(K) / K)
        self.register_buffer("mu", torch.zeros(K, ZC + 2))
        self.register_buffer("cov", torch.eye(ZC + 2).repeat(K, 1, 1))

    def features(self, x):
        zc = self.enc(x)
        xh = self.dec(zc)
        rel = (x - xh).norm(dim=1) / (x.norm(dim=1) + 1e-8)
        cos = nn.functional.cosine_similarity(x, xh, dim=1)
        z = torch.cat([zc, rel.unsqueeze(1), cos.unsqueeze(1)], dim=1)
        return xh, z

    def gmm_params(self, z, gamma):
        nk = gamma.sum(0) + 1e-8                                  # (K,)
        phi = nk / len(z)
        mu = (gamma.unsqueeze(-1) * z.unsqueeze(1)).sum(0) / nk.unsqueeze(-1)
        d = z.unsqueeze(1) - mu.unsqueeze(0)                      # (N,K,D)
        cov = (gamma.unsqueeze(-1).unsqueeze(-1)
               * d.unsqueeze(-1) @ d.unsqueeze(-2)).sum(0) / nk.view(-1, 1, 1)
        cov = cov + 1e-5 * torch.eye(z.shape[1], device=z.device)
        return phi, mu, cov

    def energy(self, z, phi, mu, cov):
        d = z.unsqueeze(1) - mu.unsqueeze(0)                      # (N,K,D)
        L = torch.linalg.cholesky(cov)                            # (K,D,D)
        sol = torch.cholesky_solve(d.transpose(0, 1).unsqueeze(-1),
                                   L.unsqueeze(1))                # (K,N,D,1)
        maha = (d.transpose(0, 1).unsqueeze(-2) @ sol).squeeze(-1).squeeze(-1)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        Dm = z.shape[1]
        logp = (phi.log().unsqueeze(1) - 0.5 * (maha + logdet.unsqueeze(1)
                + Dm * np.log(2 * np.pi)))
        return -torch.logsumexp(logp, dim=0)                      # (N,)


def main():
    set_seed()
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    Wtr = utils.make_windows(X_train, W).reshape(len(X_train) - W + 1, -1)
    Wte = utils.make_windows(X_test, W).reshape(len(X_test) - W + 1, -1)

    model = DAGMM(Wtr.shape[1]).to(DEVICE)

    def loss_fn(m, xb):
        xh, z = m.features(xb)
        gamma = m.est(z)
        phi, mu, cov = m.gmm_params(z, gamma)
        e = m.energy(z, phi, mu, cov).mean()
        covd = (1.0 / torch.diagonal(cov, dim1=-2, dim2=-1)).sum()
        return nn.functional.mse_loss(xh, xb) + LAMBDA_E * e + LAMBDA_C * covd

    fit(model, loader_from(Wtr, batch_size=256), loss_fn, epochs=20, tag=NAME)

    # freeze mixture parameters on the full healthy train set
    with torch.no_grad():
        z_all, g_all = [], []
        for i in range(0, len(Wtr), 1024):
            xb = torch.tensor(Wtr[i:i + 1024], dtype=torch.float32).to(DEVICE)
            _, z = model.features(xb)
            z_all.append(z)
            g_all.append(model.est(z))
        z_all, g_all = torch.cat(z_all), torch.cat(g_all)
        phi, mu, cov = model.gmm_params(z_all, g_all)

    def score_fn(xb):
        _, z = model.features(xb)
        return model.energy(z, phi, mu, cov)

    win_scores = batched_apply(score_fn, Wte)
    scores = utils.align_scores(win_scores, len(X_test), W)

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    win_labels = labels[W - 1:]

    def extra(ax):
        lo, hi = np.quantile(win_scores, [0.0, 0.995])
        bins = np.linspace(lo, hi, 70)
        ax.hist(np.clip(win_scores[win_labels == 0], lo, hi), bins=bins,
                color=utils.SERIES[0], alpha=0.75, label="normal windows")
        ax.hist(np.clip(win_scores[win_labels == 1], lo, hi), bins=bins,
                color=utils.CRITICAL, alpha=0.75, label="anomalous windows")
        ax.axvline(metrics["threshold"], color=utils.SERIOUS, ls="--", lw=1.2,
                   label="threshold")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("GMM sample energy -log p(z): anomalous windows sit in the "
                     "high-energy tail")
        ax.set_xlabel("energy (clipped at 99.5th pct)")

    utils.plot_detection(NAME, "DAGMM (autoencoder + GMM energy, jointly trained)",
                         test_df, labels, scores, metrics["threshold"], extra)


if __name__ == "__main__":
    main()
