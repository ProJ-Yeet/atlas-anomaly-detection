"""
09 - Variational Autoencoder (VAE)
==================================
Idea: learn a latent DISTRIBUTION representing "normal" telemetry. The encoder
maps a window to a Gaussian posterior q(z|x); the decoder maps z back to x.
Training maximises the ELBO, which pulls the posterior towards a standard
normal prior — so healthy windows occupy a smooth, dense latent region.

Anomaly score = reconstruction probability (Monte-Carlo estimate): sample z
from q(z|x) several times, average the negative log-likelihood of x under
the decoded distributions. Anomalous windows either encode far from the
prior or decode with low probability.
"""

import numpy as np
import torch
import torch.nn as nn

from common import utils
from common.torch_utils import DEVICE, batched_apply, fit, loader_from, set_seed

NAME = "09_vae"
W, ZDIM, SAMPLES = 32, 8, 8


class VAE(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU())
        self.mu, self.logvar = nn.Linear(48, ZDIM), nn.Linear(48, ZDIM)
        self.dec = nn.Sequential(nn.Linear(ZDIM, 48), nn.ReLU(), nn.Linear(48, 96),
                                 nn.ReLU(), nn.Linear(96, d_in))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-8, 4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.dec(z), mu, logvar


def main():
    set_seed()
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    Wtr = utils.make_windows(X_train, W).reshape(len(X_train) - W + 1, -1)
    Wte = utils.make_windows(X_test, W).reshape(len(X_test) - W + 1, -1)

    model = VAE(Wtr.shape[1]).to(DEVICE)

    def loss_fn(m, xb):
        xh, mu, logvar = m(xb)
        rec = ((xh - xb) ** 2).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(1).mean()
        return rec + 0.1 * kl

    fit(model, loader_from(Wtr, batch_size=256), loss_fn, epochs=20, tag=NAME)

    @torch.no_grad()
    def score_fn(xb):
        s = torch.zeros(len(xb), device=xb.device)
        for _ in range(SAMPLES):
            xh, _, _ = model(xb)
            s += ((xh - xb) ** 2).mean(1)
        return s / SAMPLES

    win_scores = batched_apply(score_fn, Wte)
    scores = utils.align_scores(win_scores, len(X_test), W)

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    # latent geometry: healthy windows hug the prior, anomalies escape it
    with torch.no_grad():
        mu_te, _ = model.encode(torch.tensor(Wte, dtype=torch.float32).to(DEVICE))
    mu_te = mu_te.cpu().numpy()
    win_labels = labels[W - 1:]

    def extra(ax):
        ax.scatter(mu_te[win_labels == 0, 0], mu_te[win_labels == 0, 1], s=3,
                   color=utils.SERIES[0], alpha=0.25, label="normal windows")
        ax.scatter(mu_te[win_labels == 1, 0], mu_te[win_labels == 1, 1], s=8,
                   color=utils.CRITICAL, label="anomalous windows")
        circ = np.linspace(0, 2 * np.pi, 100)
        ax.plot(2 * np.cos(circ), 2 * np.sin(circ), color=utils.MUTED, lw=1,
                ls=":", label="prior 2σ")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Posterior means in the first two latent dims — the prior "
                     "anchors 'normal'")
        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")

    utils.plot_detection(NAME, "VAE (reconstruction probability score)",
                         test_df, labels, scores, metrics["threshold"],
                         extra, log_score=True)


if __name__ == "__main__":
    main()
