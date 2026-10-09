"""Compare every implemented detector on one common table.

Two modes:

* **aggregate** (default) -- read the ``figures/*_metrics.json`` files written by
  each algorithm's last run and tabulate them.  Fast; run the algorithms first.
* **run** (``--run``) -- execute each algorithm's ``run()`` in-process under
  ``tracemalloc``, so the table also carries **peak memory**.  Slower (it
  retrains everything), but self-contained.

The table reports, per the brief: training time, prediction speed (rows/s),
precision, recall, F1, ROC-AUC, PR-AUC, best threshold, false positives, false
negatives, and (in ``--run`` mode) peak memory.  Results are printed, saved to
``figures/comparison.csv`` and drawn as a grouped bar chart.

Usage::

    python compare_models.py            # aggregate saved metrics
    python compare_models.py --run      # retrain everything + measure memory
    python compare_models.py --run --only isolation_forest,pca,random_forest
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
import tracemalloc
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

#: (module path, display name) for every algorithm, in difficulty order.
REGISTRY: List[tuple] = [
    ("algorithms.unsupervised.isolation_forest", "isolation_forest"),
    ("algorithms.unsupervised.dbscan", "dbscan"),
    ("algorithms.unsupervised.pca", "pca"),
    ("algorithms.unsupervised.kmeans", "kmeans"),
    ("algorithms.unsupervised.gmm", "gmm"),
    ("algorithms.supervised.random_forest", "random_forest"),
    ("algorithms.supervised.xgboost", "xgboost"),
    ("algorithms.supervised.lightgbm", "lightgbm"),
    ("algorithms.supervised.catboost", "catboost"),
    ("algorithms.supervised.logistic_regression", "logistic_regression"),
    ("algorithms.supervised.svm", "svm"),
    ("algorithms.unsupervised.autoencoder", "autoencoder"),
    ("algorithms.unsupervised.lstm_unsupervised", "lstm_unsupervised"),
    ("algorithms.unsupervised.transformer", "transformer"),
    ("algorithms.unsupervised.vae", "vae"),
    ("algorithms.unsupervised.usad", "usad"),
    ("algorithms.unsupervised.omnianomaly", "omnianomaly"),
    ("algorithms.unsupervised.gan", "gan"),
]

COLUMNS = ["model", "family", "train_time_s", "predict_time_s", "rows_per_s",
           "precision", "recall", "f1", "roc_auc", "pr_auc", "pa_f1",
           "threshold", "fp", "fn", "peak_mem_mb"]


def _row_from_metrics(name: str, m: Dict) -> Dict:
    n = m.get("n_scored", 0) or 0
    pt = m.get("predict_time_s", 0.0) or 0.0
    return {
        "model": name,
        "family": m.get("family", ""),
        "train_time_s": round(m.get("train_time_s", float("nan")), 3),
        "predict_time_s": round(pt, 3),
        "rows_per_s": round(n / pt) if pt > 0 else float("nan"),
        "precision": round(m.get("precision", float("nan")), 4),
        "recall": round(m.get("recall", float("nan")), 4),
        "f1": round(m.get("f1", float("nan")), 4),
        "roc_auc": round(m.get("roc_auc", float("nan")), 4),
        "pr_auc": round(m.get("pr_auc", float("nan")), 4),
        "pa_f1": round(m.get("pa_f1", float("nan")), 4),
        "threshold": round(m.get("threshold", float("nan")), 4),
        "fp": m.get("fp", -1),
        "fn": m.get("fn", -1),
        "peak_mem_mb": m.get("peak_mem_mb", float("nan")),
    }


def aggregate() -> pd.DataFrame:
    """Build the table from previously saved ``*_metrics.json`` files."""
    rows = []
    order = {name: i for i, (_, name) in enumerate(REGISTRY)}
    for path in glob.glob(os.path.join(FIG_DIR, "*_metrics.json")):
        name = os.path.basename(path).replace("_metrics.json", "")
        with open(path) as fh:
            rows.append(_row_from_metrics(name, json.load(fh)))
    if not rows:
        raise SystemExit("no *_metrics.json found -- run the algorithms first "
                         "(e.g. `python main.py`) or use `--run`.")
    df = pd.DataFrame(rows)
    df["_o"] = df["model"].map(lambda n: order.get(n, 999))
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def run_all(only: Optional[List[str]] = None) -> pd.DataFrame:
    """Execute each algorithm's ``run()`` and measure peak memory."""
    rows = []
    for module_path, name in REGISTRY:
        if only and name not in only:
            continue
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: import failed ({exc})")
            continue
        print(f"\n>>> running {name} ...")
        try:
            tracemalloc.start()
            metrics = mod.run()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            metrics["peak_mem_mb"] = round(peak / 1e6, 1)
            rows.append(_row_from_metrics(name, metrics))
        except SystemExit as exc:      # e.g. optional dependency missing
            tracemalloc.stop()
            print(f"[skip] {name}: {exc}")
        except Exception as exc:       # noqa: BLE001
            tracemalloc.stop()
            print(f"[error] {name}: {exc}")
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame) -> str:
    """Grouped bar chart of precision / recall / F1 / PR-AUC per model."""
    metrics = ["precision", "recall", "f1", "pr_auc"]
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.1), 6))
    x = range(len(df))
    width = 0.2
    for i, met in enumerate(metrics):
        ax.bar([xi + i * width for xi in x], df[met], width=width, label=met)
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(df["model"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Model comparison -- ATLAS DCS anomaly detection")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "comparison.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="retrain every model and measure peak memory")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated subset of model names")
    args = parser.parse_args()

    only = args.only.split(",") if args.only else None
    df = run_all(only) if args.run else aggregate()
    if df.empty:
        raise SystemExit("no results produced.")

    df = df[[c for c in COLUMNS if c in df.columns]]
    os.makedirs(FIG_DIR, exist_ok=True)
    df.to_csv(os.path.join(FIG_DIR, "comparison.csv"), index=False)

    with pd.option_context("display.width", 200,
                           "display.max_columns", None):
        print("\n" + "=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
    print(f"\nsaved figures/comparison.csv and {plot_comparison(df)}")


if __name__ == "__main__":
    main()
