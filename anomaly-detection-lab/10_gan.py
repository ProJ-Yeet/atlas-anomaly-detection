"""
10 - GAN-based anomaly detection (AnoGAN/TAnoGAN-flavoured, simplified)
=======================================================================
Idea: a Generator learns to produce "normal-looking" telemetry windows from
random noise; a Discriminator learns to tell real from generated. After
training, the Generator's range approximates the manifold of NORMAL windows
only. To score a test window we search the latent space for the z whose
G(z) best matches it (a few hundred Adam steps, batched):

  score = (1-lam) * ||x - G(z*)||   residual: can the generator imitate x?
        + lam    * ||f(x) - f(G(z*))||   discriminator feature mismatch

Anomalous windows lie OFF the learned manifold, so no z reproduces them and
the residual stays high.
"""

import numpy as np
import torch
import torch.nn as nn

from common import utils
from common.torch_utils import DEVICE, loader_from, set_seed

NAME = "10_gan"
W, ZDIM = 32, 16
EPOCHS, SEARCH_STEPS, LAM = 40, 200, 0.1


def mlp(sizes, final=None):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.LeakyReLU(0.2)]
    layers = layers[:-1] + ([final] if final else [])
    return nn.Sequential(*layers)


def main():
    set_seed()
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    d = W * X_train.shape[1]
    Wtr = utils.make_windows(X_train, W).reshape(-1, d)
    Wte = utils.make_windows(X_test, W, stride=1).reshape(-1, d)

    G = mlp([ZDIM, 64, 128, d]).to(DEVICE)
    D_feat = mlp([d, 128, 64]).to(DEVICE)          # feature extractor part
    D_head = nn.Linear(64, 1).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(list(D_feat.parameters()) + list(D_head.parameters()),
                            lr=2e-4, betas=(0.5, 0.999))
    bce = nn.functional.binary_cross_entropy_with_logits
    loader = loader_from(Wtr, batch_size=256)

    d_hist, g_hist = [], []
    for ep in range(1, EPOCHS + 1):
        dl = gl = cnt = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            z = torch.randn(len(xb), ZDIM, device=DEVICE)
            fake = G(z)

            # discriminator step (label smoothing on the real side)
            lossD = (bce(D_head(D_feat(xb)), torch.full((len(xb), 1), 0.9, device=DEVICE))
                     + bce(D_head(D_feat(fake.detach())), torch.zeros(len(xb), 1, device=DEVICE)))
            optD.zero_grad(); lossD.backward(); optD.step()

            # generator step
            lossG = bce(D_head(D_feat(fake)), torch.ones(len(xb), 1, device=DEVICE))
            optG.zero_grad(); lossG.backward(); optG.step()
            dl += lossD.item() * len(xb); gl += lossG.item() * len(xb); cnt += len(xb)
        d_hist.append(dl / cnt); g_hist.append(gl / cnt)
        if ep == 1 or ep % 10 == 0:
            print(f"  {NAME} epoch {ep:3d}/{EPOCHS}  D {d_hist[-1]:.4f}  G {g_hist[-1]:.4f}")
    G.eval(); D_feat.eval()

    # ---- AnoGAN scoring: batched latent-space search --------------------------
    x = torch.tensor(Wte, dtype=torch.float32).to(DEVICE)
    z = torch.randn(len(Wte), ZDIM, device=DEVICE, requires_grad=True)
    opt_z = torch.optim.Adam([z], lr=0.05)
    for step in range(SEARCH_STEPS):
        loss = ((G(z) - x) ** 2).mean(1).sum()
        opt_z.zero_grad(); loss.backward(); opt_z.step()
    with torch.no_grad():
        xh = G(z)
        resid = ((xh - x) ** 2).mean(1)
        featd = ((D_feat(xh) - D_feat(x)) ** 2).mean(1)
        win_scores = ((1 - LAM) * resid + LAM * featd).cpu().numpy()
        best_recon = xh.cpu().numpy()

    scores = utils.align_scores(win_scores, len(X_test), W)
    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    # panel: best generator imitation of one normal vs one anomalous window
    i_norm, i_anom = 200 - W + 1, 950 - W + 1   # healthy vs inside cascade
    steps = np.arange(W)

    def extra(ax):
        ch = 4  # cooling flow, channel hit hardest by the cascade
        ax.plot(steps, Wte[i_norm].reshape(W, -1)[:, ch], color=utils.SERIES[0],
                lw=1.2, label="normal window (actual)")
        ax.plot(steps, best_recon[i_norm].reshape(W, -1)[:, ch], color=utils.SERIES[0],
                lw=1.2, ls="--", label="best G(z) match")
        ax.plot(steps, Wte[i_anom].reshape(W, -1)[:, ch], color=utils.CRITICAL,
                lw=1.2, label="cascade window (actual)")
        ax.plot(steps, best_recon[i_anom].reshape(W, -1)[:, ch], color=utils.SERIOUS,
                lw=1.2, ls="--", label="best G(z) match")
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        ax.set_title("Latent search: G can imitate healthy cooling_flow but cannot "
                     "reach the anomalous shape")
        ax.set_xlabel("step within window")

    utils.plot_detection(NAME, "GAN (AnoGAN-style latent search score)",
                         test_df, labels, scores, metrics["threshold"], extra)


if __name__ == "__main__":
    main()
