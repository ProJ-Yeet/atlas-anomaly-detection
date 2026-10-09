"""USAD (UnSupervised Anomaly Detection) for ATLAS DCS.

*Difficulty: 12.*  One shared encoder feeds two decoders trained in a two-phase
adversarial game: AE2 learns to *amplify* the reconstruction error of AE1's
output, giving GAN-like sensitivity with autoencoder-like training stability --
from dense layers only.  The DAQ@LHC survey highlights USAD for reaching
near-transformer detection at a fraction of the compute, which is why the
project's proposal borrows its scoring.

How it works (epoch-dependent weight n = current epoch)
-------------------------------------------------------
* AE1: minimise  (1/n)||x-AE1(x)||^2 + (1-1/n)||x-AE2(AE1(x))||^2
  -> reconstruct x *and* make the output AE2-proof.
* AE2: minimise  (1/n)||x-AE2(x)||^2 - (1-1/n)||x-AE2(AE1(x))||^2
  -> reconstruct x but *amplify* AE1's residual.

Early epochs are plain autoencoding (1/n ~ 1); as n grows the adversarial terms
dominate.  Score = alpha*||x-AE1(x)||^2 + beta*||x-AE2(AE1(x))||^2, with alpha/beta
trading false positives against sensitivity *at inference time without
retraining* -- a genuinely useful operational dial.

Run standalone::

    python algorithms/unsupervised/usad.py
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
from torch.nn.utils import clip_grad_norm_  # noqa: E402

from utils.torch_utils import (DEVICE, batch_iter, finalize_sequences,  # noqa: E402
                               prepare_sequences, set_seed)

NAME = "usad"


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, latent), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, input_dim: int, latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, input_dim))

    def forward(self, z):
        return self.net(z)


class USAD(nn.Module):
    """Shared encoder E with two decoders D1, D2."""

    def __init__(self, input_dim: int, latent: int = 20) -> None:
        super().__init__()
        self.E = _Encoder(input_dim, latent)
        self.D1 = _Decoder(input_dim, latent)
        self.D2 = _Decoder(input_dim, latent)

    def w1(self, x):
        return self.D1(self.E(x))

    def w2(self, x):
        return self.D2(self.E(x))

    def w2_of_w1(self, x):
        return self.D2(self.E(self.w1(x)))


def run(window: int = 30, epochs: int = 12, latent: int = 20,
        alpha: float = 0.5, beta: float = 0.5, seed: int = 42) -> dict:
    """Train USAD adversarially; score with the alpha/beta combination."""
    set_seed(seed)
    seq = prepare_sequences(window=window, train_stride=3, seed=seed)
    input_dim = seq.window * seq.n_features
    Xtr = seq.X_train.reshape(len(seq.X_train), -1)
    Xall = seq.X_all.reshape(len(seq.X_all), -1)
    print(f"[{NAME}] windows: train={len(Xtr):,} all={len(Xall):,} "
          f"input_dim={input_dim}")

    model = USAD(input_dim, latent).to(DEVICE)
    # Two optimizers, both including the shared encoder E (as in the official
    # USAD). The stability trick is to compute *one* forward pass, back-propagate
    # BOTH losses, then step both optimizers together -- never recompute the
    # forward on half-updated weights (doing so is what inverts/diverges the
    # score). Gradient clipping is kept as a safety net.
    opt1 = torch.optim.Adam(list(model.E.parameters()) +
                            list(model.D1.parameters()), lr=1e-3)
    opt2 = torch.optim.Adam(list(model.E.parameters()) +
                            list(model.D2.parameters()), lr=1e-3)
    mse = nn.MSELoss()

    # --- two-phase adversarial training ----------------------------------- #
    t0 = time.perf_counter()
    model.train()
    for ep in range(1, epochs + 1):
        n = float(ep)
        rw = 1.0 / n            # reconstruction weight (USAD 1/n schedule)
        aw = 1.0 - rw           # adversarial weight
        l1_acc = l2_acc = 0.0
        for (xb,) in batch_iter(Xtr, batch=256, shuffle=True, seed=ep):
            z = model.E(xb)
            w1 = model.D1(z)
            w2 = model.D2(z)
            w3 = model.D2(model.E(w1))          # AE2 applied to AE1's output
            loss1 = rw * mse(w1, xb) + aw * mse(w3, xb)
            loss2 = rw * mse(w2, xb) - aw * mse(w3, xb)

            opt1.zero_grad(); opt2.zero_grad()
            loss1.backward(retain_graph=True)
            loss2.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            opt1.step(); opt2.step()
            l1_acc += float(loss1); l2_acc += float(loss2)
        print(f"    epoch {ep:>2}/{epochs}  loss1={l1_acc:.2f}  "
              f"loss2={l2_acc:.2f}")
    train_time = time.perf_counter() - t0

    # --- score ------------------------------------------------------------ #
    t1 = time.perf_counter()
    model.eval()
    scores = np.zeros(len(Xall), dtype=np.float32)
    with torch.no_grad():
        pos = 0
        for (xb,) in batch_iter(Xall, batch=512, shuffle=False):
            w1 = model.w1(xb)
            s = (alpha * ((xb - w1) ** 2).mean(dim=1)
                 + beta * ((xb - model.D2(model.E(w1))) ** 2).mean(dim=1))
            scores[pos:pos + len(xb)] = s.cpu().numpy(); pos += len(xb)
    predict_time = time.perf_counter() - t1

    return finalize_sequences(NAME, seq, scores, train_time, predict_time)


if __name__ == "__main__":
    run()
