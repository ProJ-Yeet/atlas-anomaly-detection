"""Data loading, splitting and scaling for the ATLAS DCS pipeline.

This module is the single entry point every algorithm script uses to obtain
data, so the loading / scaling conventions stay identical across all 17
detectors (any difference in results is then a difference between *algorithms*,
not between their harnesses).

Key conventions
---------------
* **Long format.** One row per (element, timestamp).  ``row_id`` is a stable
  global index used to join predictions back to ground truth.
* **Temporal split.** Train/test splits are taken *per element in time order*
  (the first ``train_frac`` of each element's series is "train"), never a
  random shuffle -- shuffling a time series leaks the future into the past.
* **Unsupervised honesty.** The unlabeled loader returns telemetry with **no**
  label column; ground truth is loaded separately and used only for scoring.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PKG_ROOT, "data")
LABELED_CSV = os.path.join(DATA_DIR, "supervised", "dcs_labeled.csv")
UNLABELED_CSV = os.path.join(DATA_DIR, "unsupervised", "dcs_unlabeled.csv")
GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "unsupervised", "ground_truth.csv")

#: Raw sensor channels present in the dataset.
SENSOR_COLS: List[str] = [
    "temperature", "pressure", "voltage", "current",
    "error_rate", "heartbeat", "quality_flag", "alarm",
]

#: Non-feature bookkeeping columns.
META_COLS: List[str] = ["row_id", "timestamp", "element_id", "subsystem"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_dataset.py` first.")


@lru_cache(maxsize=None)
def _read_sorted(path: str) -> pd.DataFrame:
    """Parse + sort one telemetry CSV, once per process.

    The 18 detectors each re-read the same one or two CSVs; parsing dominates
    the cost, so the parsed frame is cached and every ``load_*`` hands back a
    defensive copy (a copy is ~10x cheaper than a re-parse and keeps callers
    free to mutate what they are given).
    """
    _require(path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["element_id", "timestamp"]).reset_index(drop=True)


@lru_cache(maxsize=None)
def _read_plain(path: str) -> pd.DataFrame:
    """Parse one CSV that needs no date parsing or sorting, once per process."""
    _require(path)
    return pd.read_csv(path)


def load_labeled() -> pd.DataFrame:
    """Load the labeled dataset (includes ``anomaly`` and ``anomaly_type``)."""
    return _read_sorted(LABELED_CSV).copy()


def load_unlabeled() -> pd.DataFrame:
    """Load the unlabeled dataset (telemetry only, no label columns)."""
    return _read_sorted(UNLABELED_CSV).copy()


def load_ground_truth() -> pd.DataFrame:
    """Load ``row_id,anomaly,anomaly_type`` for scoring unsupervised models."""
    return _read_plain(GROUND_TRUTH_CSV).copy()


def attach_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join ground-truth labels onto an unlabeled frame by ``row_id``.

    Used *only* for evaluation of unsupervised models, never for training.
    """
    gt = load_ground_truth()
    return df.merge(gt, on="row_id", how="left")


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def temporal_split(df: pd.DataFrame, train_frac: float = 0.6,
                   group_col: str = "element_id"
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split each element's series in time order into (train, test).

    The first ``train_frac`` of every element's timeline goes to train, the
    remainder to test.  Both partitions therefore cover every element and every
    subsystem, but never share a timestamp -- the honest setup for time series.
    """
    train_parts, test_parts = [], []
    for _, g in df.groupby(group_col, sort=False):
        g = g.sort_values("timestamp")
        cut = int(len(g) * train_frac)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


# --------------------------------------------------------------------------- #
# Feature-matrix helpers
# --------------------------------------------------------------------------- #

#: Columns kept in the frame for reference/baselines but NEVER fed to a model.
#: ``alarm`` is the detector's own threshold-alarm decision -- using it as an
#: input to another anomaly detector is circular (it partly encodes "something
#: is wrong"), so it and its derived features are held out and used only as the
#: naive baseline to beat. ``quality_flag`` (SCADA data-validity) is a genuine,
#: independent input and is retained.
LEAKY_OR_CIRCULAR: List[str] = ["alarm", "time_since_alarm", "alarm_freq"]


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Numeric feature columns = telemetry only, excluding metadata, labels,
    the raw ``heartbeat`` counter, and the circular ``alarm``-derived signals.

    * ``anomaly`` / ``anomaly_type`` -- ground-truth labels, never inputs.
    * ``heartbeat`` -- a wrapping 0-255 byte with no meaningful magnitude; its
      engineered ``heartbeat_delta`` (0 == frozen) carries the signal instead.
    * ``alarm`` / ``time_since_alarm`` / ``alarm_freq`` -- the detector's own
      alarm decision and its history; held out so models learn from raw physics
      telemetry, and ``alarm`` serves as the baseline detector instead.
    """
    drop = (set(META_COLS) | {"anomaly", "anomaly_type", "heartbeat"}
            | set(LEAKY_OR_CIRCULAR))
    return [c for c in df.columns
            if c not in drop and np.issubdtype(df[c].dtype, np.number)]

def get_matrix(df: pd.DataFrame, cols: Optional[List[str]] = None
               ) -> np.ndarray:
    """Return a float matrix of the requested feature columns (NaN -> 0)."""
    cols = cols or feature_columns(df)
    return df[cols].to_numpy(dtype=float, na_value=0.0)


def standardize(train_X: np.ndarray, test_X: Optional[np.ndarray] = None,
                ) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Z-score using **train** statistics only (no leakage from test)."""
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_X)
    test_s = scaler.transform(test_X) if test_X is not None else None
    return train_s, test_s, scaler
