"""
06 - USAD (UnSupervised Anomaly Detection), Audibert et al. 2020
================================================================
Idea: two autoencoders share one encoder E and are trained ADVERSARIALLY:

  AE1 = D1(E(x)),  AE2 = D2(E(x)),  and the two-pass output AE2(AE1(x)).

Phase-in over epochs (n = epoch number):
  loss1 = (1/n)||x - AE1(x)||^2 + (1 - 1/n)||x - AE2(AE1(x))||^2   (fool AE2)
  loss2 = (1/n)||x - AE2(x)||^2 - (1 - 1/n)||x - AE2(AE1(x))||^2   (detect AE1)

AE2 learns to amplify any reconstruction imperfection, which makes the
combined score alpha*||x-AE1(x)|| + beta*||x-AE2(AE1(x))|| far more sensitive
than a single AE, while staying cheap (dense layers only). This efficiency is
why the DAQ@LHC review found USAD attractive for hybrid setups like DeepHYDRA.
"""

import torch
import torch.nn as nn

from common import utils
from common.torch_utils import DEVICE, batched_apply, loader_from, set_seed

NAME = "06_usad"
W, LATENT = 32, 16
EPOCHS, ALPHA, BETA = 30, 0.5, 0.5


def mlp(sizes, final=None):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers = layers[:-1] + ([final] if final else [])
    return nn.Sequential(*layers)


class USAD(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.E = mlp([d, d // 2, LATENT])
        self.D1 = mlp([LATENT, d // 2, d])
        self.D2 = mlp([LATENT, d // 2, d])


def main():
    set_seed()
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    Wtr = utils.make_windows(X_train, W).reshape(len(X_train) - W + 1, -1)
    Wte = utils.make_windows(X_test, W).reshape(len(X_test) - W + 1, -1)

    model = USAD(Wtr.shape[1]).to(DEVICE)
    opt1 = torch.optim.Adam(list(model.E.parameters()) + list(model.D1.parameters()), lr=1e-3)
    opt2 = torch.optim.Adam(list(model.E.parameters()) + list(model.D2.parameters()), lr=1e-3)
    loader = loader_from(Wtr, batch_size=256)
    mse = nn.functional.mse_loss

    hist1, hist2 = [], []
    for ep in range(1, EPOCHS + 1):
        t1 = t2 = cnt = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            k = 1.0 / ep
            w1 = model.D1(model.E(xb))
            w3 = model.D2(model.E(w1))
            loss1 = k * mse(w1, xb) + (1 - k) * mse(w3, xb)
            opt1.zero_grad(); loss1.backward(); opt1.step()

            w1 = model.D1(model.E(xb)).detach()
            w2 = model.D2(model.E(xb))
            w3 = model.D2(model.E(w1))
            loss2 = k * mse(w2, xb) - (1 - k) * mse(w3, xb)
            opt2.zero_grad(); loss2.backward(); opt2.step()
            t1 += loss1.item() * len(xb); t2 += loss2.item() * len(xb); cnt += len(xb)
        hist1.append(t1 / cnt); hist2.append(t2 / cnt)
        if ep == 1 or ep % 5 == 0:
            print(f"  {NAME} epoch {ep:3d}/{EPOCHS}  loss1 {hist1[-1]:.5f}  loss2 {hist2[-1]:.5f}")
    model.eval()

    @torch.no_grad()
    def score_fn(xb):
        w1 = model.D1(model.E(xb))
        w3 = model.D2(model.E(w1))
        return (ALPHA * ((xb - w1) ** 2).mean(1)
                + BETA * ((xb - w3) ** 2).mean(1))

    win_scores = batched_apply(score_fn, Wte)
    scores = utils.align_scores(win_scores, len(X_test), W)

    metrics = utils.evaluate(scores, labels)
    utils.save_result(NAME, metrics)

    def extra(ax):
        ax.plot(range(1, EPOCHS + 1), hist1, color=utils.SERIES[0], lw=1.4,
                label="loss1 (AE1 tries to fool AE2)")
        ax.plot(range(1, EPOCHS + 1), hist2, color=utils.SERIES[2], lw=1.4,
                label="loss2 (AE2 learns to expose AE1)")
        ax.axhline(0, color=utils.MUTED, lw=0.8)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Adversarial two-phase training — loss2 goes negative as AE2 "
                     "learns to amplify reconstruction defects")
        ax.set_xlabel("epoch")

    utils.plot_detection(NAME, "USAD (adversarially trained twin autoencoders)",
                         test_df, labels, scores, metrics["threshold"],
                         extra, log_score=True)


if __name__ == "__main__":
    main()
