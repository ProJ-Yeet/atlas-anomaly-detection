"""OmniAnomaly-inspired stochastic recurrent detector for ATLAS DCS.

*Difficulty: 13.*  A simplified take on OmniAnomaly: model normality as a
*distribution over trajectories* rather than a single reconstruction.  A GRU
carries temporal state; at each step the model infers a Gaussian posterior over
a latent z and a decoder maps z to a Gaussian over the observation (mean **and**
variance).  Training maximises the ELBO; the anomaly score is the negative
reconstruction log-likelihood of the window's last step.

Why the learned variance matters
--------------------------------
A plain autoencoder pays the same penalty for noise it cannot predict as for a
real anomaly.  A probabilistic decoder learns "this channel is noisy, +/-0.2 is
normal" and prices deviations in *units of learned uncertainty* -- which is why
OmniAnomaly typically wins precision on heterogeneous SCADA-like channels, the
exact character of DCS telemetry.

Simplifications vs. the paper: standard-normal prior (no linear Gaussian
state-space prior), no planar normalizing flows.  This keeps it runnable on CPU
while preserving the core stochastic-recurrent idea.

Run standalone::

    python algorithms/unsupervised/omnianomaly.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from utils.torch_utils import (DEVICE, batch_iter, finalize_sequences,  # noqa: E402
                               prepare_sequences, set_seed)

NAME = "omnianomaly"


class OmniAnomaly(nn.Module):
    """GRU encoder -> Gaussian latent -> GRU decoder -> Gaussian observations."""

    def __init__(self, n_features: int, hidden: int = 48,
                 latent: int = 12) -> None:
        super().__init__()
        self.enc_gru = nn.GRU(n_features, hidden, batch_first=True)
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.dec_gru = nn.GRU(latent, hidden, batch_first=True)
        self.out_mu = nn.Linear(hidden, n_features)
        self.out_logvar = nn.Linear(hidden, n_features)

    def forward(self, x):  # noqa: D401
        """Return (x_mu, x_logvar, z_mu, z_logvar) for the whole window."""
        h, _ = self.enc_gru(x)
        z_mu, z_logvar = self.to_mu(h), self.to_logvar(h)
        z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mu)
        d, _ = self.dec_gru(z)
        return self.out_mu(d), self.out_logvar(d), z_mu, z_logvar


def _gaussian_nll(x, mu, logvar):
    """Per-element negative log-likelihood of x under N(mu, exp(logvar))."""
    return 0.5 * (logvar + (x - mu) ** 2 / torch.exp(logvar))


def run(window: int = 30, epochs: int = 6, hidden: int = 48, latent: int = 12,
        beta: float = 0.5, seed: int = 42) -> dict:
    """Train the stochastic-recurrent model; score by last-step NLL."""
    set_seed(seed)
    seq = prepare_sequences(window=window, train_stride=4, seed=seed)
    print(f"[{NAME}] windows: train={len(seq.X_train):,} "
          f"all={len(seq.X_all):,}  n_features={seq.n_features}")

    model = OmniAnomaly(seq.n_features, hidden, latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- train (maximise ELBO over the window) ---------------------------- #
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        total, nb = 0.0, 0
        for (xb,) in batch_iter(seq.X_train, batch=256, shuffle=True, seed=ep):
            opt.zero_grad()
            x_mu, x_logvar, z_mu, z_logvar = model(xb)
            recon = _gaussian_nll(xb, x_mu, x_logvar).mean()
            kl = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            loss = recon + beta * kl
            loss.backward(); opt.step()
            total += float(loss) * len(xb); nb += len(xb)
        print(f"    epoch {ep + 1:>2}/{epochs}  elbo_loss={total / nb:.5f}")
    train_time = time.perf_counter() - t0

    # --- score: negative log-likelihood of the LAST step ------------------ #
    t1 = time.perf_counter()
    model.eval()
    scores = np.zeros(len(seq.X_all), dtype=np.float32)
    with torch.no_grad():
        pos = 0
        for (xb,) in batch_iter(seq.X_all, batch=512, shuffle=False):
            x_mu, x_logvar, _, _ = model(xb)
            nll_last = _gaussian_nll(xb[:, -1], x_mu[:, -1],
                                     x_logvar[:, -1]).mean(dim=1)
            scores[pos:pos + len(xb)] = nll_last.cpu().numpy(); pos += len(xb)
    predict_time = time.perf_counter() - t1

    return finalize_sequences(NAME, seq, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
