"""
11 - Adaptive thresholds via Extreme Value Theory: POT and SPOT
===============================================================
Every detector in this lab outputs a SCORE; turning scores into alarms needs a
threshold, and static thresholds break on non-stationary telemetry. Extreme
Value Theory gives a principled alternative:

  POT (Peak Over Threshold): pick a modest initial level u (e.g. the 98th
  percentile), fit a Generalized Pareto Distribution to the excesses above u,
  then compute the level z_q exceeded with a chosen tiny probability q.
  The alarm threshold is derived from a target FALSE-ALARM RATE, not eyeballed.

  SPOT (Streaming POT): same, but processed online — every new excess updates
  the GPD fit, so the threshold ADAPTS as the score distribution drifts.

Score stream used here: k-NN distance to healthy train states (same signal
as 01_dbscan) — deliberately simple, to keep the focus on the thresholding.
"""

import numpy as np
from scipy.stats import genpareto

from common import utils

NAME = "11_pot_adaptive_threshold"
Q = 1e-3          # target exceedance probability
U_QUANTILE = 0.98  # initial level for excess collection


def pot_threshold(scores, q=Q, u_quantile=U_QUANTILE):
    """Classic POT: GPD fit on excesses over u -> level z_q."""
    u = np.quantile(scores, u_quantile)
    excesses = scores[scores > u] - u
    xi, _, sigma = genpareto.fit(excesses, floc=0)
    n, nt = len(scores), len(excesses)
    if abs(xi) < 1e-6:
        zq = u + sigma * np.log(nt / (q * n))
    else:
        zq = u + (sigma / xi) * ((q * n / nt) ** (-xi) - 1)
    return zq, u, xi, sigma, excesses


def spot_stream(scores, calibration, q=Q):
    """SPOT: initial POT on the calibration set, then streaming updates."""
    zq, u, xi, sigma, _ = pot_threshold(calibration, q)
    excesses = list(calibration[calibration > u] - u)
    n = len(calibration)
    thresholds, alarms = np.empty(len(scores)), np.zeros(len(scores), dtype=bool)
    for t, s in enumerate(scores):
        thresholds[t] = zq
        if s > zq:
            alarms[t] = True          # alarm: do NOT absorb into the model
        elif s > u:
            excesses.append(s - u)    # new tail sample -> refit
            n += 1
            xi, _, sigma = genpareto.fit(excesses, floc=0)
            nt = len(excesses)
            zq = (u + sigma * np.log(nt / (q * n)) if abs(xi) < 1e-6
                  else u + (sigma / xi) * ((q * n / nt) ** (-xi) - 1))
        else:
            n += 1
    return thresholds, alarms


def main():
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)

    calib, scores = utils.knn_score(X_train, X_test)   # healthy calibration + test

    thresholds, alarms = spot_stream(scores, calib)
    preds = alarms.astype(int)
    p, r, f1 = utils._prf(preds, labels)
    pa_p, pa_r, pa_f1 = utils._prf(utils.point_adjust(preds, labels), labels)
    zq0, u, xi, sigma, excesses = pot_threshold(calib)
    utils.save_result(NAME, {
        "threshold": float(np.median(thresholds)), "precision": p, "recall": r,
        "f1": f1, "pa_precision": pa_p, "pa_recall": pa_r, "pa_f1": pa_f1,
        "gpd_xi": float(xi), "gpd_sigma": float(sigma), "initial_zq": float(zq0),
        "per_fault": utils.fault_coverage(preds)})
    print(f"[{NAME}] F1={f1:.3f} P={p:.3f} R={r:.3f} | PA-F1={pa_f1:.3f}")

    def extra(ax):
        # GPD tail fit quality on the calibration excesses
        xs = np.linspace(0, excesses.max() * 1.1, 200)
        ax.hist(excesses, bins=50, density=True, color=utils.SERIES[0],
                alpha=0.7, label="calibration excesses over u")
        ax.plot(xs, genpareto.pdf(xs, xi, 0, sigma), color=utils.SERIES[2],
                lw=1.6, label=f"GPD fit (ξ={xi:.2f}, σ={sigma:.3f})")
        ax.axvline(zq0 - u, color=utils.SERIOUS, ls="--", lw=1.2,
                   label=f"z_q - u for q={Q:g}")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Extreme Value Theory: the fitted tail converts a target "
                     "false-alarm rate into a threshold")
        ax.set_xlabel("excess above u")

    # custom score panel showing the STREAMING threshold
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.6), constrained_layout=True)
    fig.suptitle("Adaptive Thresholding: POT / SPOT (Extreme Value Theory)",
                 fontsize=12, fontweight="bold", color=utils.INK)
    Xz = (test_df - test_df.mean()) / test_df.std()
    ax = axes[0]
    for i, ch in enumerate(Xz.columns):
        ax.plot(Xz[ch] + 4 * (len(Xz.columns) - 1 - i), color=utils.SERIES[i], lw=0.7)
        ax.text(len(Xz) + 30, 4 * (len(Xz.columns) - 1 - i), ch,
                color=utils.SERIES[i], va="center", fontsize=8)
    utils._shade_truth(ax, labels)
    ax.set_yticks([]); ax.set_xlim(0, len(Xz) * 1.14)
    ax.set_title("Test telemetry (z-scored, offset) — true anomalies shaded")

    ax = axes[1]
    ax.plot(scores, color=utils.INK_2, lw=0.8, label="k-NN distance score")
    ax.plot(thresholds, color=utils.SERIOUS, lw=1.4,
            label="SPOT threshold (adapts online)")
    ax.axhline(np.quantile(calib, 0.999), color=utils.MUTED, ls=":", lw=1.2,
               label="static 99.9th-pct threshold")
    det = np.flatnonzero(alarms)
    ax.plot(det, scores[det], ".", color=utils.CRITICAL, ms=4, label="SPOT alarms")
    utils._shade_truth(ax, labels)
    ax.set_yscale("log"); ax.set_xlim(0, len(scores) * 1.14)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Streaming threshold vs static threshold")
    ax.set_xlabel("time step")

    extra(axes[2])
    import os
    out = os.path.join(utils.FIG_DIR, f"{NAME}.png")
    os.makedirs(utils.FIG_DIR, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[{NAME}] figure -> figures/{NAME}.png")


if __name__ == "__main__":
    main()
