"""
07 - OmniAnomaly (simplified), Su et al. 2019
=============================================
Idea: a STOCHASTIC recurrent model. A GRU carries the temporal state; at every
time step the model infers a DISTRIBUTION over a latent z_t (not a point!),
samples from it, and reconstructs x_t. Training maximises the ELBO
(reconstruction log-likelihood minus KL to the prior). The anomaly score is
the negative reconstruction log-probability: for healthy telemetry the model
is confident (high log-prob), for anomalous inputs the probability collapses.

Because normality is modelled as a probability distribution rather than a
single reconstruction, OmniAnomaly is robust to the natural noise of sensor
data. (Simplified here: standard-normal prior instead of the paper's linear
Gaussian state-space model + planar normalizing flows.)
"""

import numpy as np
import torch
import torch.nn as nn

from common import utils
from common.torch_utils import DEVICE, batched_apply, fit, loader_from, set_seed

NAME = "07_omnianomaly"
W, HIDDEN, ZDIM = 32, 48, 4


class OmniLite(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.rnn = nn.GRU(d, HIDDEN, batch_first=True)
        self.z_mu = nn.Linear(HIDDEN, ZDIM)
        self.z_logvar = nn.Linear(HIDDEN, ZDIM)
        self.dec = nn.GRU(ZDIM, HIDDEN, batch_first=True)
        self.x_mu = nn.Linear(HIDDEN, d)
        self.x_logvar = nn.Linear(HIDDEN, d)

    def forward(self, x):
        h, _ = self.rnn(x)                                   # (B,T,H)
        mu, logvar = self.z_mu(h), self.z_logvar(h).clamp(-8, 4)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() # reparameterise
        hd, _ = self.dec(z)
        xmu, xlogvar = self.x_mu(hd), self.x_logvar(hd).clamp(-8, 4)
        return xmu, xlogvar, mu, logvar

    @staticmethod
    def nll(x, xmu, xlogvar):
        """Per-step negative Gaussian log-likelihood, summed over channels."""
        return 0.5 * (xlogvar + (x - xmu) ** 2 / xlogvar.exp()
                      + np.log(2 * np.pi)).sum(-1)           # (B,T)


def main():
    set_seed()
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    Wtr = utils.make_windows(X_train, W)
    Wte = utils.make_windows(X_test, W)

    model = OmniLite(X_train.shape[1]).to(DEVICE)

    def loss_fn(m, xb):
        xmu, xlogvar, mu, logvar = m(xb)
        rec = m.nll(xb, xmu, xlogvar).mean()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(-1).mean()
        return rec + 0.05 * kl

    fit(model, loader_from(Wtr), loss_fn, epochs=15, tag=NAME)

    @torch.no_grad()
    def score_fn(xb):
        # average NLL of the LAST step over a few posterior samples
        s = torch.zeros(len(xb), device=xb.device)
        for _ in range(4):
            xmu, xlogvar, _, _ = model(xb)
            s += model.nll(xb, xmu, xlogvar)[:, -1]
        return s / 4

    win_scores = batched_apply(score_fn, Wte)
    scores = utils.align_scores(win_scores, len(X_test), W)

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    # show predictive distribution vs actual around the cascade (800-1100)
    seg = slice(700, 1200)
    with torch.no_grad():
        xb = torch.tensor(Wte[seg.start - W + 1:seg.stop - W + 1],
                          dtype=torch.float32).to(DEVICE)
        xmu, xlogvar, _, _ = model(xb)
        mu_last = xmu[:, -1, 0].cpu().numpy()
        sd_last = (0.5 * xlogvar[:, -1, 0]).exp().cpu().numpy()
    idx = np.arange(seg.start, seg.stop)

    def extra(ax):
        ax.plot(idx, X_test[seg, 0], color=utils.SERIES[0], lw=1.0,
                label="actual pixel_temp (z)")
        ax.plot(idx, mu_last, color=utils.SERIES[4], lw=1.0, ls="--",
                label="reconstruction mean")
        ax.fill_between(idx, mu_last - 2 * sd_last, mu_last + 2 * sd_last,
                        color=utils.SERIES[4], alpha=0.18, lw=0,
                        label="±2σ model belief")
        ax.axvspan(800, 1100, color=utils.CRITICAL, alpha=0.13, lw=0)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Probabilistic reconstruction — during the cascade the actual "
                     "signal leaves the model's belief band")
        ax.set_xlabel("time step")

    utils.plot_detection(NAME, "OmniAnomaly-lite (stochastic GRU-VAE, NLL score)",
                         test_df, labels, scores, metrics["threshold"], extra,
                         log_score=True)


if __name__ == "__main__":
    main()
