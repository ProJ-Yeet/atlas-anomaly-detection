"""Variational Autoencoder anomaly detection for ATLAS DCS.

*Difficulty: 11.*  Like a dense autoencoder, but the encoder outputs a
*distribution* q(z|x) = N(mu, sigma) and the loss adds a KL term pulling that
posterior toward a standard-normal prior.  The KL regularisation forbids the
latent space from having "holes", so low reconstruction probability becomes a
*trustworthy* anomaly signal rather than an artifact of a ragged code layout.

How it works
------------
Encoder -> (mu, log-var); sample z via the reparameterisation trick; decoder ->
reconstruction.  Loss = reconstruction MSE + beta * KL(q||N(0,I)).  The anomaly
score is the Monte-Carlo reconstruction error over several posterior samples
(the reconstruction-probability estimator of An & Cho).

Why it fits ATLAS DCS
---------------------
The prior gives a natural "distance from normality" geometry and lets you sample
synthetic healthy windows for testing.  It is the static (window) ancestor of
OmniAnomaly -- comparing the two isolates what recurrence adds.  As a
non-recurrent window model it shares USAD's contextual-fault ceiling.

Run standalone::

    python algorithms/unsupervised/vae.py
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

NAME = "vae"


class VAE(nn.Module):
    """Fully-connected VAE over a flattened window."""

    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, input_dim))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):  # noqa: D401
        """Return (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        return self.dec(self.reparameterise(mu, logvar)), mu, logvar


def run(window: int = 30, epochs: int = 8, latent_dim: int = 16,
        beta: float = 0.5, mc_samples: int = 5, seed: int = 42) -> dict:
    """Train the VAE (ELBO) and score by MC reconstruction error."""
    set_seed(seed)
    seq = prepare_sequences(window=window, train_stride=3, seed=seed)
    input_dim = seq.window * seq.n_features
    Xtr = seq.X_train.reshape(len(seq.X_train), -1)
    Xall = seq.X_all.reshape(len(seq.X_all), -1)
    print(f"[{NAME}] windows: train={len(Xtr):,} all={len(Xall):,} "
          f"input_dim={input_dim}")

    model = VAE(input_dim, latent_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- train (maximise the ELBO) ---------------------------------------- #
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        total, nb = 0.0, 0
        for (xb,) in batch_iter(Xtr, batch=256, shuffle=True, seed=ep):
            opt.zero_grad()
            rec, mu, logvar = model(xb)
            recon = ((rec - xb) ** 2).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl
            loss.backward(); opt.step()
            total += float(loss) * len(xb); nb += len(xb)
        print(f"    epoch {ep + 1:>2}/{epochs}  elbo_loss={total / nb:.5f}")
    train_time = time.perf_counter() - t0

    # --- score: mean reconstruction error over MC posterior samples ------- #
    t1 = time.perf_counter()
    model.eval()
    scores = np.zeros(len(Xall), dtype=np.float32)
    with torch.no_grad():
        pos = 0
        for (xb,) in batch_iter(Xall, batch=512, shuffle=False):
            acc = torch.zeros(len(xb), device=DEVICE)
            for _ in range(mc_samples):
                rec, _, _ = model(xb)
                acc += ((rec - xb) ** 2).mean(dim=1)
            scores[pos:pos + len(xb)] = (acc / mc_samples).cpu().numpy()
            pos += len(xb)
    predict_time = time.perf_counter() - t1

    return finalize_sequences(NAME, seq, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
