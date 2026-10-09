"""Dense Autoencoder anomaly detection for ATLAS DCS.

*Difficulty: 8 (first deep model).*  A neural network trained to reconstruct its
input through a narrow bottleneck.  Trained on (mostly-healthy) windows, it
learns to rebuild normal telemetry accurately; an anomalous window cannot be
reconstructed as well, so the **reconstruction error** is the anomaly score.

How it works
------------
Each 30-step x n-feature window is flattened to a vector; an encoder compresses
it to a small latent code and a decoder expands it back.  The bottleneck forces
the network to keep only the dominant (normal) structure.  This is the nonlinear
generalisation of PCA reconstruction -- swap the linear projection for a learned
nonlinear one.

Why it fits ATLAS DCS
---------------------
Unsupervised (no labels needed), captures nonlinear channel couplings PCA
misses, and after training is a pure ``score(window) -> float`` function that
shards across Spark executors via ``TorchDistributor``.  A flattened dense AE has
no temporal inductive bias, so it dilutes short point-spikes across the window --
the LSTM/Transformer variants address that.

Run standalone::

    python algorithms/unsupervised/autoencoder.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from torch import nn  # noqa: E402

from utils.torch_utils import (fit_reconstruction, finalize_sequences,  # noqa: E402
                               prepare_sequences, reconstruction_scores,
                               set_seed)

NAME = "autoencoder"


class DenseAutoencoder(nn.Module):
    """Symmetric fully-connected autoencoder over a flattened window."""

    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):  # noqa: D401
        """Reconstruct the flattened window."""
        return self.decoder(self.encoder(x))


def run(window: int = 30, epochs: int = 8, latent_dim: int = 16,
        seed: int = 42) -> dict:
    """Train the dense AE on strided windows; score every window."""
    set_seed(seed)
    seq = prepare_sequences(window=window, train_stride=3, seed=seed)
    input_dim = seq.window * seq.n_features
    print(f"[{NAME}] windows: train={len(seq.X_train):,} all={len(seq.X_all):,}"
          f"  input_dim={input_dim}")

    model = DenseAutoencoder(input_dim, latent_dim)
    t0 = time.perf_counter()
    fit_reconstruction(model, seq.X_train, epochs=epochs, flatten=True)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    win_scores = reconstruction_scores(model, seq.X_all, flatten=True)
    predict_time = time.perf_counter() - t1

    return finalize_sequences(NAME, seq, win_scores, train_time, predict_time)


if __name__ == "__main__":
    run()
