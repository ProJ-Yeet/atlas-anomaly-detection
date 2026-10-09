"""GAN-based anomaly detection for ATLAS DCS.

*Difficulty: 14 (hardest / least reliable).*  A generator G and discriminator D
play the usual minimax game until G's range approximates the manifold of
*healthy* windows (it has never seen anything else).  The discriminator learns
to tell real healthy windows from generated ones; at inference, an anomalous
window is off the healthy manifold, so D's "realness" estimate for it is low.

Scoring (simplified AnoGAN)
---------------------------
The canonical AnoGAN inverts the generator per window (optimise z to minimise
||G(z)-x||) -- accurate but expensive (an optimisation loop *per window*), a poor
match for high-throughput DCS scoring.  This prototype instead uses a
**discriminator feature-matching** score: the anomaly score is a blend of D's
negative logit and the L2 distance in D's penultimate feature space between x and
its best cheap generator match.  It is fast (one forward pass) at the cost of the
accuracy the full latent search would add.

Honest expectation
-------------------
GANs are the weakest detector in this family on small data (training is
unstable; mode coverage is incomplete) -- included for completeness and because
the survey lists it.  In practice GANs are better used to *augment* rare-fault
data than as the primary detector.

Run standalone::

    python algorithms/unsupervised/gan.py
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

NAME = "gan"


class Generator(nn.Module):
    def __init__(self, latent: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, output_dim))

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Discriminator whose penultimate layer doubles as a feature extractor."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
                                      nn.Linear(128, 64), nn.LeakyReLU(0.2))
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f), f


def run(window: int = 30, epochs: int = 15, latent: int = 20,
        seed: int = 42) -> dict:
    """Train a GAN on healthy windows; score by discriminator feature matching."""
    set_seed(seed)
    seq = prepare_sequences(window=window, train_stride=3, seed=seed)
    input_dim = seq.window * seq.n_features
    Xtr = seq.X_train.reshape(len(seq.X_train), -1)
    Xall = seq.X_all.reshape(len(seq.X_all), -1)
    print(f"[{NAME}] windows: train={len(Xtr):,} all={len(Xall):,} "
          f"input_dim={input_dim}")

    G = Generator(latent, input_dim).to(DEVICE)
    D = Discriminator(input_dim).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    # --- adversarial training --------------------------------------------- #
    t0 = time.perf_counter()
    for ep in range(epochs):
        dl = gl = 0.0
        for (xb,) in batch_iter(Xtr, batch=256, shuffle=True, seed=ep):
            bs = len(xb)
            real = torch.ones(bs, 1, device=DEVICE)
            fake = torch.zeros(bs, 1, device=DEVICE)
            # train D
            z = torch.randn(bs, latent, device=DEVICE)
            optD.zero_grad()
            real_logit, _ = D(xb)
            fake_logit, _ = D(G(z).detach())
            loss_d = bce(real_logit, real) + bce(fake_logit, fake)
            loss_d.backward(); optD.step()
            # train G
            optG.zero_grad()
            z = torch.randn(bs, latent, device=DEVICE)
            gen_logit, _ = D(G(z))
            loss_g = bce(gen_logit, real)
            loss_g.backward(); optG.step()
            dl += float(loss_d); gl += float(loss_g)
        print(f"    epoch {ep + 1:>2}/{epochs}  loss_D={dl:.2f} loss_G={gl:.2f}")
    train_time = time.perf_counter() - t0

    # --- score: (1-lambda)*(-D logit) + lambda*feature residual ----------- #
    # Cheap generator match: a batch of random z, keep the nearest G(z) in
    # feature space (a single-shot stand-in for AnoGAN's per-window search).
    t1 = time.perf_counter()
    G.eval(); D.eval()
    lam = 0.5
    scores = np.zeros(len(Xall), dtype=np.float32)
    with torch.no_grad():
        z_bank = torch.randn(512, latent, device=DEVICE)
        gen_bank = G(z_bank)
        _, gen_feats = D(gen_bank)                 # (512, 64)
        pos = 0
        for (xb,) in batch_iter(Xall, batch=512, shuffle=False):
            logit, feats = D(xb)                   # (B,1), (B,64)
            # nearest generated feature vector per row
            d2 = torch.cdist(feats, gen_feats)     # (B, 512)
            residual = d2.min(dim=1).values
            realness = torch.sigmoid(logit).squeeze(1)
            s = (1 - lam) * (1 - realness) + lam * (residual / residual.mean())
            scores[pos:pos + len(xb)] = s.cpu().numpy(); pos += len(xb)
    predict_time = time.perf_counter() - t1

    return finalize_sequences(NAME, seq, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
