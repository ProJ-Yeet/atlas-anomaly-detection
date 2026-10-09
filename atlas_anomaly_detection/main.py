"""Orchestrator for the ATLAS DCS anomaly-detection prototype.

Runs the whole pipeline end to end:

1. (Re)generate the synthetic dataset if it is missing.
2. Run every implemented detector (or a chosen subset / family).
3. Build the cross-model comparison table and figure.

Usage::

    python main.py                       # generate (if needed) + run all + compare
    python main.py --family unsupervised # only the unsupervised detectors
    python main.py --only isolation_forest,pca,random_forest
    python main.py --skip-deep           # skip the slow deep-learning models
    python main.py --regenerate          # force-regenerate the dataset first
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_models import REGISTRY, aggregate, plot_comparison  # noqa: E402

DEEP = {"autoencoder", "lstm_unsupervised", "transformer", "vae", "usad",
        "omnianomaly", "gan"}


def ensure_dataset(regenerate: bool = False) -> None:
    """Generate the dataset if the CSVs are absent (or if forced)."""
    from utils.preprocessing import LABELED_CSV
    if regenerate or not os.path.exists(LABELED_CSV):
        print("[main] generating synthetic dataset ...")
        from data.generate_dataset import generate, write_datasets
        write_datasets(generate())
    else:
        print("[main] dataset present -- skipping generation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["supervised", "unsupervised"],
                        default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    ensure_dataset(args.regenerate)

    only = set(args.only.split(",")) if args.only else None
    ran = 0
    for module_path, name in REGISTRY:
        family = "supervised" if ".supervised." in module_path else \
                 "unsupervised"
        if only and name not in only:
            continue
        if args.family and family != args.family:
            continue
        if args.skip_deep and name in DEEP:
            continue
        print(f"\n{'=' * 70}\n[main] {name}\n{'=' * 70}")
        try:
            importlib.import_module(module_path).run()
            ran += 1
        except SystemExit as exc:
            print(f"[main] skipped {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[main] ERROR in {name}: {exc}")

    print(f"\n[main] ran {ran} model(s). Building comparison ...")
    try:
        df = aggregate()
        df.to_csv(os.path.join(os.path.dirname(__file__), "figures",
                               "comparison.csv"), index=False)
        print(df.to_string(index=False))
        plot_comparison(df)
    except SystemExit as exc:
        print(f"[main] comparison skipped: {exc}")


if __name__ == "__main__":
    main()
