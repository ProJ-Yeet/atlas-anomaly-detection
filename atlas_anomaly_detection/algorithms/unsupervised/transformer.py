"""Transformer Autoencoder anomaly detection for ATLAS DCS.

*Difficulty: 10.*  A self-attention encoder that reconstructs a window.  Unlike
the LSTM, self-attention relates *every* pair of timesteps in one parallel
operation, so it captures long-range temporal dependencies without sequential
recurrence -- and it is GPU-friendly at scale.

How it works
------------
Each timestep's feature vector is linearly embedded and given a learned
positional encoding (attention is order-agnostic without it).  A stack of
Transformer encoder blocks (multi-head self-attention + feed-forward) produces a
contextualised representation that a linear head maps back to the original
channels.  Input dropout acts as a **denoising objective**, preventing the model
from learning a trivial identity copy and forcing it to encode real structure.
Reconstruction error is the anomaly score.

Why it fits ATLAS DCS
---------------------
Parallel across the window (GPU-efficient), strong on collective/frequency
faults (attention locks onto periodic structure), and interpretable via
attention maps (property g).  For short 30-step windows its O(T^2) advantage is
idle; it earns its keep when windows grow to hours/days of context.

Run standalone::

    python algorithms/unsupervised/transformer.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from utils.torch_utils import (FEW_THREADS, fit_reconstruction,  # noqa: E402
                               finalize_sequences, prepare_sequences,
                               reconstruction_scores, set_seed, torch_threads)

NAME = "transformer"


class TransformerAutoencoder(nn.Module):
    """Pre-norm Transformer encoder that reconstructs a window (denoising)."""

    def __init__(self, n_features: int, window: int, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        self.in_drop = nn.Dropout(dropout)     # denoising: corrupt the input
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_features)

    def forward(self, x):  # noqa: D401
        """Reconstruct the window from its (noised) self-attention encoding."""
        h = self.embed(self.in_drop(x)) + self.pos
        h = self.encoder(h)
        return self.head(h)


def run(window: int = 30, epochs: int = 6, d_model: int = 32,
        seed: int = 42) -> dict:
    """Train the Transformer AE on strided windows; score every window."""
    set_seed(seed)
    seq = prepare_sequences(window=window, train_stride=4, seed=seed)
    print(f"[{NAME}] windows: train={len(seq.X_train):,} all={len(seq.X_all):,}"
          f"  n_features={seq.n_features}")

    model = TransformerAutoencoder(seq.n_features, window=window,
                                   d_model=d_model)
    # Self-attention over a 30-step window is a long chain of *small* matmuls,
    # so intra-op threading spends more on barriers than it recovers: measured
    # 1.43x faster on 2 threads than on torch's default 4 (see torch_threads).
    with torch_threads(FEW_THREADS):
        t0 = time.perf_counter()
        fit_reconstruction(model, seq.X_train, epochs=epochs, batch=256,
                           flatten=False)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        win_scores = reconstruction_scores(model, seq.X_all, flatten=False)
        predict_time = time.perf_counter() - t1

    return finalize_sequences(NAME, seq, win_scores, train_time, predict_time)


if __name__ == "__main__":
    run()
