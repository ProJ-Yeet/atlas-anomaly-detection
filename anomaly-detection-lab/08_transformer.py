"""
08 - Transformer-based anomaly detection (TranAD-flavoured, simplified)
=======================================================================
Idea: self-attention lets every time step attend to every other step in the
window directly (no recurrence), so long-range temporal dependencies are
modelled in a single hop and windows are processed in parallel. We train a
small transformer encoder to reconstruct masked windows of healthy telemetry;
the reconstruction error is the anomaly score. The attention map itself is a
useful diagnostic: for anomalous windows the heads concentrate on the
offending steps.
"""

import numpy as np
import torch
import torch.nn as nn

from common import utils
from common.torch_utils import DEVICE, batched_apply, fit, loader_from, set_seed

NAME = "08_transformer"
W, D_MODEL, HEADS, LAYERS = 32, 32, 4, 2


class Block(nn.Module):
    """Pre-norm transformer block that exposes its attention weights."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, 64), nn.GELU(),
                                nn.Linear(64, D_MODEL))
        self.n1, self.n2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.last_attn = None

    def forward(self, x):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        self.last_attn = w.detach()
        x = x + a
        return x + self.ff(self.n2(x))


class TransformerAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inp = nn.Linear(d, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, W, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.out = nn.Linear(D_MODEL, d)

    def forward(self, x):
        h = self.inp(x) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def main():
    set_seed()
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    Wtr = utils.make_windows(X_train, W)
    Wte = utils.make_windows(X_test, W)

    model = TransformerAE(X_train.shape[1]).to(DEVICE)

    def loss_fn(m, xb):
        # denoising objective: randomly mask 20% of steps, reconstruct all
        mask = (torch.rand(xb.shape[0], W, 1, device=xb.device) < 0.2).float()
        return nn.functional.mse_loss(m(xb * (1 - mask)), xb)

    fit(model, loader_from(Wtr), loss_fn, epochs=15, tag=NAME)

    win_scores = batched_apply(
        lambda xb: ((model(xb) - xb) ** 2).mean(dim=(1, 2)), Wte)
    scores = utils.align_scores(win_scores, len(X_test), W)

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    # attention map for a window ending inside the HV oscillation burst (A4)
    probe = 2700
    with torch.no_grad():
        xb = torch.tensor(Wte[probe - W + 1][None], dtype=torch.float32).to(DEVICE)
        model(xb)
        attn = model.blocks[-1].last_attn[0].cpu().numpy()

    def extra(ax):
        im = ax.imshow(attn, aspect="auto", cmap="Blues", origin="lower")
        ax.figure.colorbar(im, ax=ax, shrink=0.85, label="attention weight")
        ax.set_title(f"Self-attention map (last layer, window ending at t={probe}, "
                     "inside the HV oscillation burst)")
        ax.set_xlabel("attended step (key)")
        ax.set_ylabel("query step")
        ax.grid(False)

    utils.plot_detection(NAME, "Transformer autoencoder (masked reconstruction)",
                         test_df, labels, scores, metrics["threshold"],
                         extra, log_score=True)


if __name__ == "__main__":
    main()
