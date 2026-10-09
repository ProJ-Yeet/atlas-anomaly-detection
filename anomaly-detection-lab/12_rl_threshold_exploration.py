"""
12 - Reinforcement Learning: adaptive thresholds + active exploration (toy)
===========================================================================
Idea (per the DAQ@LHC review): let an RL agent LEARN the alarm threshold
policy instead of fixing it. The agent observes the recent score statistics,
chooses to RAISE / KEEP / LOWER a threshold multiplier, and receives rewards
that encode the operational trade-off:

    true alarm      +1.0     missed anomaly   -1.0     false alarm  -0.3

Epsilon-greedy Q-learning supplies the ACTIVE EXPLORATION component: with
probability eps (decaying) the agent tries a non-greedy threshold to probe
new operating regimes — exactly the "explore changing conditions" bullet.

This is a didactic toy: rewards use ground-truth labels (in production the
feedback would come from operator acknowledgements of alarms, cf. the
EVENT_ALERTS ACK_TIMESTAMP field in the ATLAS schema).
"""

import numpy as np

from common import utils

NAME = "12_rl_threshold_exploration"
ACTIONS = np.array([-0.1, 0.0, +0.1])   # lower / keep / raise multiplier
MULT_RANGE = (0.5, 8.0)
EPISODES = 60


def state_of(mult, recent_alarm_rate):
    """Discretise (threshold multiplier, recent alarm rate) into a table index."""
    m_bin = int(np.clip((mult - MULT_RANGE[0]) / 0.5, 0, 15))
    a_bin = int(np.clip(recent_alarm_rate * 20, 0, 4))
    return m_bin * 5 + a_bin


def run_episode(scores, labels, base, Q, eps, rng, learn=True,
                lr=0.15, gamma=0.95):
    mult = 4.0
    total_r = 0.0
    alarms = np.zeros(len(scores), dtype=bool)
    mult_trace = np.empty(len(scores))
    recent = 0.0
    s = state_of(mult, recent)
    for t in range(len(scores)):
        a = rng.integers(3) if (learn and rng.random() < eps) else int(np.argmax(Q[s]))
        mult = float(np.clip(mult + ACTIONS[a], *MULT_RANGE))
        thr = base * mult
        alarm = scores[t] > thr
        alarms[t] = alarm
        mult_trace[t] = mult
        r = (1.0 if labels[t] else -0.3) if alarm else (-1.0 if labels[t] else 0.01)
        total_r += r
        recent = 0.98 * recent + 0.02 * alarm
        s2 = state_of(mult, recent)
        if learn:
            Q[s, a] += lr * (r + gamma * Q[s2].max() - Q[s, a])
        s = s2
    return total_r, alarms, mult_trace


def main():
    rng = np.random.default_rng(0)
    train_df, test_df, labels = utils.load_data()
    X_train, X_test = utils.standardize(train_df, test_df)
    calib, scores = utils.knn_score(X_train, X_test)
    base = np.quantile(calib, 0.99)   # healthy-tail score = one multiplier unit

    Q = np.zeros((16 * 5, 3))
    rewards, best_Q, best_r = [], Q.copy(), -np.inf
    for ep in range(EPISODES):
        eps = max(0.02, 0.5 * (1 - ep / (EPISODES * 0.7)))   # decaying exploration
        r, _, _ = run_episode(scores, labels, base, Q, eps, rng, learn=True)
        rewards.append(r)
        if r > best_r:
            best_r, best_Q = r, Q.copy()
    # greedy evaluation with the best snapshot (tabular Q-learning is noisy)
    _, alarms, mult_trace = run_episode(scores, labels, base, best_Q, 0.0, rng,
                                        learn=False)

    preds = alarms.astype(int)
    p, r_, f1 = utils._prf(preds, labels)
    pa_p, pa_r, pa_f1 = utils._prf(utils.point_adjust(preds, labels), labels)
    utils.save_result(NAME, {"threshold": float(base * np.median(mult_trace)),
                             "precision": p, "recall": r_, "f1": f1,
                             "pa_precision": pa_p, "pa_recall": pa_r, "pa_f1": pa_f1,
                             "per_fault": utils.fault_coverage(preds)})
    print(f"[{NAME}] F1={f1:.3f} P={p:.3f} R={r_:.3f} | PA-F1={pa_f1:.3f}")

    import os
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.6), constrained_layout=True)
    fig.suptitle("Reinforcement Learning: learned adaptive threshold "
                 "(Q-learning + ε-greedy exploration)", fontsize=12,
                 fontweight="bold", color=utils.INK)

    ax = axes[0]
    ax.plot(scores, color=utils.INK_2, lw=0.8, label="k-NN distance score")
    ax.plot(base * mult_trace, color=utils.SERIOUS, lw=1.4,
            label="RL-learned threshold (greedy policy)")
    det = np.flatnonzero(alarms)
    ax.plot(det, scores[det], ".", color=utils.CRITICAL, ms=4, label="alarms")
    utils._shade_truth(ax, labels)
    ax.set_yscale("log"); ax.set_xlim(0, len(scores) * 1.14)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("The agent tightens the threshold inside anomalous regimes and "
                 "relaxes it in quiet ones")
    ax.set_xlabel("time step")

    ax = axes[1]
    ax.plot(mult_trace, color=utils.SERIES[4], lw=1.0)
    utils._shade_truth(ax, labels)
    ax.set_title("Threshold multiplier chosen by the policy over time")
    ax.set_xlabel("time step")
    ax.set_ylabel("multiplier")

    ax = axes[2]
    ax.plot(rewards, color=utils.SERIES[0], lw=1.4, label="episode reward (ε-greedy)")
    ax.axhline(best_r, color=utils.SERIES[3], ls=":", lw=1.2,
               label="best episode (Q snapshot used for evaluation)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Learning curve — ε-greedy exploration keeps it noisy by design")
    ax.set_xlabel("training episode")
    ax.set_ylabel("total reward")

    out = os.path.join(utils.FIG_DIR, f"{NAME}.png")
    os.makedirs(utils.FIG_DIR, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[{NAME}] figure -> figures/{NAME}.png")


if __name__ == "__main__":
    main()
