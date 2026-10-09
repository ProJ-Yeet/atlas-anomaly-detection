"""LSTM Autoencoder anomaly detection for ATLAS DCS.

*Difficulty: 9.*  A sequence-to-sequence autoencoder whose encoder and decoder
are LSTMs, so it reconstructs a window while explicitly modelling **temporal
order** -- what follows what, at which phase, with which cross-channel lag.

How it works
------------
The encoder LSTM reads the window step by step and ends with a hidden state
summarising the whole sequence.  That single state is repeated across time and
fed to a decoder LSTM that must regenerate the entire window (no teacher
forcing, so the bottleneck is real).  Reconstruction error is the anomaly score.
Because the model internalises the temporal grammar of normal operation, it
flags *contextual* faults (right value, wrong time) that per-point methods miss.

Why it fits ATLAS DCS
---------------------
Directly matches cascading, lagged multi-channel faults (cooling -> temp -> HV
with a ~30-step lag is learnable) and contextual anomalies.  This is the deep
half of the proposal's DeepHYDRA-style hybrid.  Cost: sequential computation (no
parallelism across timesteps), so it is the slowest family per window and wants
a GPU at full scale.

Run standalone::

    python algorithms/unsupervised/lstm_unsupervised.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from utils.torch_utils import (fit_reconstruction, finalize_sequences,  # noqa: E402
                               prepare_sequences, reconstruction_scores,
                               set_seed)

NAME = "lstm_unsupervised"


class LSTMAutoencoder(nn.Module):
    """Encoder-decoder LSTM that reconstructs a (window, n_feat) sequence."""

    def __init__(self, n_features: int, hidden: int = 48,
                 window: int = 30) -> None:
        super().__init__()
        self.window = window
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x):  # noqa: D401
        """Return the reconstructed window (same shape as ``x``)."""
        _, (h, _) = self.encoder(x)                 # h: (1, B, hidden)
        z = h[-1].unsqueeze(1).repeat(1, self.window, 1)  # (B, window, hidden)
        dec, _ = self.decoder(z)
        return self.head(dec)


def run(window: int = 30, epochs: int = 6, hidden: int = 48,
        seed: int = 42) -> dict:
    """Train the LSTM-AE on strided windows; score every window."""
    set_seed(seed)
    seq = prepare_sequences(window=window, train_stride=4, seed=seed)
    print(f"[{NAME}] windows: train={len(seq.X_train):,} all={len(seq.X_all):,}"
          f"  n_features={seq.n_features}")

    model = LSTMAutoencoder(seq.n_features, hidden=hidden, window=window)
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
