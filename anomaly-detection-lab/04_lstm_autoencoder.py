"""
04 - LSTM Autoencoder
=====================
Idea: an LSTM encoder reads a window step-by-step and compresses it into a
single hidden state; an LSTM decoder must rebuild the whole window from that
state alone. Trained on healthy telemetry only, the model learns the normal
temporal grammar (daily cycles, temp<->HV coupling). At test time the
reconstruction error is the anomaly score: sequences the model has never
seen — including CONTEXTUAL anomalies whose values are in-range but whose
timing is wrong — reconstruct badly.

This is the deep half of the DeepHYDRA-style hybrid in the project proposal.
"""

import numpy as np
import torch
import torch.nn as nn

from common import utils
from common.data_generator import CHANNELS
from common.torch_utils import DEVICE, batched_apply, fit, loader_from, set_seed

NAME = "04_lstm_autoencoder"
W, HIDDEN = 32, 48


class LSTMAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.LSTM(d, HIDDEN, batch_first=True)
        self.dec = nn.LSTM(d, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, d)

    def forward(self, x):
        _, (h, c) = self.enc(x)
        # decode from the compressed state, feeding zeros (teacher-free)
        dec_in = torch.zeros_like(x)
        y, _ = self.dec(dec_in, (h, c))
        return self.out(y)


def main():
    set_seed()
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    Wtr = utils.make_windows(X_train, W)
    Wte = utils.make_windows(X_test, W)

    model = LSTMAE(X_train.shape[1]).to(DEVICE)
    fit(model, loader_from(Wtr),
        lambda m, xb: nn.functional.mse_loss(m(xb), xb),
        epochs=15, tag=NAME)

    win_scores = batched_apply(
        lambda xb: ((model(xb) - xb) ** 2).mean(dim=(1, 2)), Wte)
    scores = utils.align_scores(win_scores, len(X_test), W)

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    # reconstruction detail around the contextual anomaly (1900-2100)
    seg = slice(1800, 2200)
    idx = np.arange(seg.start, seg.stop)
    with torch.no_grad():
        xb = torch.tensor(Wte[seg.start - W + 1:seg.stop - W + 1],
                          dtype=torch.float32).to(DEVICE)
        recon_last = model(xb)[:, -1, :].cpu().numpy()

    def extra(ax):
        ch = 0  # pixel temperature
        ax.plot(idx, X_test[seg, ch], color=utils.SERIES[0], lw=1.0,
                label="actual pixel_temp (z)")
        ax.plot(idx, recon_last[:, ch], color=utils.SERIES[4], lw=1.0, ls="--",
                label="LSTM-AE reconstruction")
        ax.axvspan(1900, 2100, color=utils.CRITICAL, alpha=0.13, lw=0)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Contextual anomaly: values in range, but the model expects "
                     "the healthy phase -> reconstruction gap")
        ax.set_xlabel("time step")

    utils.plot_detection(NAME, "LSTM Autoencoder (reconstruction error)",
                         test_df, labels, scores, metrics["threshold"],
                         extra, log_score=True)


if __name__ == "__main__":
    main()
