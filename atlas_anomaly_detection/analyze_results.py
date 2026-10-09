"""Per-anomaly-type and model-complementarity analysis.

Reads the ``figures/*_predictions.csv`` written by every detector plus the
ground-truth types, and answers three questions the raw comparison table cannot:

1. **Which algorithm is best at which anomaly TYPE?**  -> per-type detection
   coverage (recall) heatmap + a best-model-per-type chart.
2. **How do the algorithms differ?**  -> a complementarity (error-correlation)
   heatmap: low correlation == the two models catch *different* anomalies.
3. **Which algorithms work best TOGETHER?**  -> a greedy union-coverage curve
   (set-cover) and a table of curated OR / majority-vote ensembles with their
   precision / recall / F1.

Everything is computed on a **common test split** (last 40% of each element's
timeline) so supervised and unsupervised models are compared on the same rows.
A naive ``alarm``-bit baseline is included so the ML lift is explicit.

Outputs (in ``figures/``): ``anomaly_type_coverage.csv`` +
``anomaly_type_coverage.png``, ``best_algo_per_type.png``,
``model_complementarity.png``, ``ensemble_greedy_coverage.png``,
``ensemble_table.csv``.

Run after the detectors (``python main.py``)::

    python analyze_results.py
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compare_models import REGISTRY  # noqa: E402
from utils.preprocessing import load_labeled, temporal_split  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _all_rows() -> pd.DataFrame:
    """Full labeled frame: row_id, anomaly, anomaly_type, alarm (all 100k)."""
    df = load_labeled()
    return df[["row_id", "anomaly", "anomaly_type", "alarm"]].copy()


def _test_rows() -> pd.DataFrame:
    """Return the common test split with row_id, anomaly, anomaly_type, alarm."""
    df = load_labeled()
    _, test = temporal_split(df, train_frac=0.6)
    return test[["row_id", "anomaly", "anomaly_type", "alarm"]].copy()


def load_predictions_full(all_rows: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Each model's predictions over ALL the rows it scored, with types.

    Unsupervised models score every row (all 11 anomaly types visible);
    supervised models score only their held-out test rows (8 types). Used for
    the per-type coverage table, which is about *type* completeness rather than
    row-for-row comparability.
    """
    types = all_rows.set_index("row_id")["anomaly_type"]
    out: Dict[str, pd.DataFrame] = {}
    for _, name in REGISTRY:
        path = os.path.join(FIG_DIR, f"{name}_predictions.csv")
        if not os.path.exists(path):
            continue
        p = pd.read_csv(path, usecols=["row_id", "anomaly", "predicted"])
        p["anomaly_type"] = p["row_id"].map(types)
        out[name] = p
    return out


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Precision, recall, F1 for the positive class (no sklearn dependency)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def load_predictions(test: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Load each model's predictions restricted to the common test rows."""
    test_ids = set(test["row_id"])
    types = test.set_index("row_id")["anomaly_type"]
    out: Dict[str, pd.DataFrame] = {}
    for _, name in REGISTRY:
        path = os.path.join(FIG_DIR, f"{name}_predictions.csv")
        if not os.path.exists(path):
            continue
        p = pd.read_csv(path, usecols=["row_id", "anomaly", "predicted"])
        p = p[p["row_id"].isin(test_ids)].copy()
        p["anomaly_type"] = p["row_id"].map(types)
        out[name] = p
    return out


# --------------------------------------------------------------------------- #
# 1. Per-type coverage
# --------------------------------------------------------------------------- #

def per_type_coverage(preds: Dict[str, pd.DataFrame], all_rows: pd.DataFrame
                      ) -> pd.DataFrame:
    """Recall of each model on each anomaly type (fraction of that type flagged).

    Computed over every row each model scored, so all anomaly types present in
    the dataset appear (a model that never saw a given type shows NaN)."""
    types = sorted(t for t in all_rows["anomaly_type"].unique() if t != "normal")
    rows = {}
    # naive alarm baseline (over all rows)
    base = {}
    for t in types:
        m = all_rows["anomaly_type"] == t
        base[t] = float(all_rows.loc[m, "alarm"].mean()) if m.any() else np.nan
    rows["alarm_baseline"] = base
    for name, p in preds.items():
        cov = {}
        for t in types:
            m = p["anomaly_type"] == t
            cov[t] = float(p.loc[m, "predicted"].mean()) if m.any() else np.nan
        rows[name] = cov
    return pd.DataFrame(rows).T[types]


def plot_coverage_heatmap(cov: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(max(9, len(cov.columns) * 1.0),
                                    max(6, len(cov) * 0.45)))
    im = ax.imshow(cov.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(len(cov.columns)))
    ax.set_xticklabels(cov.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cov.index)))
    ax.set_yticklabels(cov.index, fontsize=8)
    for i in range(len(cov.index)):
        for j in range(len(cov.columns)):
            v = cov.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    ax.set_title("Detection coverage (recall) by anomaly type")
    fig.colorbar(im, ax=ax, fraction=0.025, label="fraction of type flagged")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "anomaly_type_coverage.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out


def plot_best_per_type(cov: pd.DataFrame) -> str:
    models_only = cov.drop(index="alarm_baseline", errors="ignore")
    best_model = models_only.idxmax(axis=0)
    best_val = models_only.max(axis=0)
    fig, ax = plt.subplots(figsize=(max(9, len(cov.columns) * 1.1), 5))
    bars = ax.bar(range(len(cov.columns)), best_val.values, color="#4e79a7")
    ax.set_xticks(range(len(cov.columns)))
    ax.set_xticklabels(cov.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("best coverage (recall)")
    ax.set_title("Best algorithm per anomaly type")
    for i, (bar, m) in enumerate(zip(bars, best_model.values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, m,
                rotation=90, ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "best_algo_per_type.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 2. Complementarity
# --------------------------------------------------------------------------- #

def catch_matrix(preds: Dict[str, pd.DataFrame], test: pd.DataFrame
                 ) -> pd.DataFrame:
    """Binary matrix: rows = models, cols = true anomalous test rows, 1 == caught."""
    anom = test[test["anomaly"] == 1]["row_id"].to_numpy()
    mat = {}
    for name, p in preds.items():
        s = p.set_index("row_id")["predicted"]
        mat[name] = s.reindex(anom).fillna(0).to_numpy().astype(int)
    return pd.DataFrame(mat, index=anom).T


def plot_complementarity(catch: pd.DataFrame) -> str:
    corr = np.corrcoef(catch.to_numpy())
    names = list(catch.index)
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7),
                                    max(7, len(names) * 0.7)))
    im = ax.imshow(corr, cmap="RdYlGn_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Detection-agreement correlation\n(green = complementary, "
                 "catch different anomalies)")
    fig.colorbar(im, ax=ax, fraction=0.045, label="correlation of catches")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "model_complementarity.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 3. Best-together: greedy set cover + curated ensembles
# --------------------------------------------------------------------------- #

def greedy_cover(catch: pd.DataFrame) -> List[tuple]:
    """Greedy union coverage: repeatedly add the model catching the most
    still-missed anomalies. Returns [(model, cumulative_recall), ...]."""
    total = catch.shape[1]
    remaining = np.ones(total, dtype=bool)
    order = []
    pool = set(catch.index)
    caught_running = np.zeros(total, dtype=bool)
    while pool:
        best, best_gain, best_vec = None, -1, None
        for name in pool:
            vec = catch.loc[name].to_numpy().astype(bool)
            gain = int((vec & remaining).sum())
            if gain > best_gain:
                best, best_gain, best_vec = name, gain, vec
        caught_running = caught_running | best_vec
        remaining = remaining & ~best_vec
        order.append((best, caught_running.sum() / total))
        pool.discard(best)
        if best_gain == 0:
            break
    return order


def plot_greedy(order: List[tuple]) -> str:
    names = [o[0] for o in order]
    cov = [o[1] for o in order]
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.6), 5))
    ax.plot(range(1, len(cov) + 1), cov, "-o", color="#59a14f")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("cumulative recall (union)"); ax.set_ylim(0, 1.02)
    ax.set_title("Greedy ensemble: marginal anomaly coverage added per model")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ensemble_greedy_coverage.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out


def ensemble_table(preds: Dict[str, pd.DataFrame], test: pd.DataFrame,
                   combos: Dict[str, tuple]) -> pd.DataFrame:
    """Precision/recall/F1 of OR-combined and majority-vote ensembles."""
    y_true = test.set_index("row_id")["anomaly"]
    rows = []
    # per-model preds aligned to test row order
    aligned = {n: p.set_index("row_id")["predicted"].reindex(y_true.index)
               .fillna(0).astype(int) for n, p in preds.items()}
    yt = y_true.to_numpy()

    # single best F1 model for reference
    for name, series in aligned.items():
        p, r, f1 = _prf(yt, series.to_numpy())
        rows.append(["(single) " + name, p, r, f1])

    for label, members in combos.items():
        members = [m for m in members if m in aligned]
        if not members:
            continue
        stack = np.vstack([aligned[m].to_numpy() for m in members])
        or_pred = (stack.sum(axis=0) >= 1).astype(int)
        maj_pred = (stack.sum(axis=0) >= (len(members) + 1) // 2).astype(int)
        p, r, f1 = _prf(yt, or_pred)
        rows.append([f"OR[{label}]", p, r, f1])
        if len(members) >= 3:
            p2, r2, f2 = _prf(yt, maj_pred)
            rows.append([f"MAJORITY[{label}]", p2, r2, f2])

    df = pd.DataFrame(rows, columns=["ensemble", "precision", "recall", "f1"])
    return df.sort_values("f1", ascending=False).reset_index(drop=True)


def main() -> None:
    if not glob.glob(os.path.join(FIG_DIR, "*_predictions.csv")):
        raise SystemExit("no predictions found -- run `python main.py` first.")
    all_rows = _all_rows()
    test = _test_rows()
    preds = load_predictions(test)              # common rows for fair combining
    preds_full = load_predictions_full(all_rows)  # all rows for type coverage
    print(f"[analyze] {len(preds)} models | per-type over all rows, "
          f"ensembles over {len(test):,} common test rows "
          f"({int(test['anomaly'].sum()):,} anomalous)")

    # 1. per-type coverage
    cov = per_type_coverage(preds_full, all_rows)
    cov.round(3).to_csv(os.path.join(FIG_DIR, "anomaly_type_coverage.csv"))
    print("\n=== Detection coverage (recall) by anomaly type ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(cov.round(2).to_string())
    plot_coverage_heatmap(cov)
    plot_best_per_type(cov)

    print("\n=== Best algorithm per anomaly type ===")
    models_only = cov.drop(index="alarm_baseline", errors="ignore")
    for t in cov.columns:
        print(f"  {t:<24s} -> {models_only[t].idxmax():<20s} "
              f"(recall {models_only[t].max():.2f}); "
              f"alarm baseline {cov.loc['alarm_baseline', t]:.2f}")

    # 2. complementarity
    catch = catch_matrix(preds, test)
    plot_complementarity(catch)

    # 3. best-together
    order = greedy_cover(catch)
    plot_greedy(order)
    print("\n=== Greedy ensemble (marginal coverage) ===")
    for i, (name, c) in enumerate(order, 1):
        print(f"  {i:>2}. + {name:<20s} cumulative recall = {c:.3f}")

    combos = {
        "IF+OmniAnomaly": ("isolation_forest", "omnianomaly"),
        "IF+LSTM": ("isolation_forest", "lstm_unsupervised"),
        "hybrid(IF+LSTM+Omni)": ("isolation_forest", "lstm_unsupervised",
                                 "omnianomaly"),
        "top_supervised(RF+XGB+LGBM)": ("random_forest", "xgboost", "lightgbm"),
        "spatial(IF+PCA+GMM)": ("isolation_forest", "pca", "gmm"),
        "all_unsupervised": tuple(n for _, n in REGISTRY
                                  if "unsupervised" in _),
    }
    etab = ensemble_table(preds, test, combos)
    etab.round(4).to_csv(os.path.join(FIG_DIR, "ensemble_table.csv"),
                         index=False)
    print("\n=== Ensembles: precision / recall / F1 (top 15 by F1) ===")
    print(etab.head(15).round(3).to_string(index=False))
    print(f"\nsaved: anomaly_type_coverage.[csv,png], best_algo_per_type.png, "
          f"model_complementarity.png, ensemble_greedy_coverage.png, "
          f"ensemble_table.csv  (all in figures/)")


if __name__ == "__main__":
    main()
