"""Sliding-window construction for the sequence models.

Deep temporal detectors (LSTM-AE, Transformer, USAD, OmniAnomaly, ...) score a
*window* of consecutive timesteps rather than a single row.  Windows must never
straddle two elements, so they are cut **within each ``element_id``** in
timestamp order.

The convention here is "score the last step of the window": each window's label
and its position (``row_id`` / row index) are taken from its final timestep, so
a window's anomaly score can be written back to exactly one row.

Windows are cut with one vectorised gather per call rather than a Python loop
over ~100k windows: :func:`window_gather_index` builds the (n_windows, window)
index matrix from per-element arithmetic, and the tensor is then materialised as
a single ``values[gather]`` fancy-index straight into float32.  This avoids the
float64 ``np.stack`` of 100k small arrays the loop version built (which peaked
at ~3x the final tensor's memory).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def window_gather_index(df: pd.DataFrame, window: int = 30, stride: int = 1,
                        group_col: str = "element_id") -> np.ndarray:
    """Positional row indices of every window, shape ``(n_windows, window)``.

    Row ``i`` holds the ``window`` positional indices into ``df`` (which must
    already carry a clean 0..n-1 index) that make up window ``i``.  Windows are
    cut inside each group and never cross a group boundary; groups appear in
    first-seen order, matching ``groupby(sort=False)``.
    """
    parts = []
    offsets = np.arange(window)
    for _, g in df.groupby(group_col, sort=False):
        gpos = np.asarray(g.index)
        n = len(gpos)
        if n < window:
            continue
        starts = np.arange(0, n - window + 1, stride)
        parts.append(gpos[starts[:, None] + offsets])
    if not parts:
        return np.empty((0, window), dtype=np.int64)
    return np.concatenate(parts, axis=0)


def stride_subset(df: pd.DataFrame, window: int = 30, stride: int = 1,
                  group_col: str = "element_id") -> np.ndarray:
    """Positions of the stride-``stride`` windows inside the stride-1 window list.

    ``make_windows(df, ..., stride=k)`` returns exactly
    ``make_windows(df, ..., stride=1)[stride_subset(df, window, k)]``.  Lets a
    caller build the dense (stride-1) tensor once and slice the strided training
    subset out of it instead of walking the frame a second time.
    """
    parts, offset = [], 0
    for _, g in df.groupby(group_col, sort=False):
        n = len(g)
        if n < window:
            continue
        n_windows = n - window + 1
        parts.append(offset + np.arange(0, n_windows, stride))
        offset += n_windows
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def make_windows(df: pd.DataFrame, feature_cols: List[str], window: int = 30,
                 stride: int = 1, group_col: str = "element_id",
                 label_col: Optional[str] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Build sliding windows within each group.

    Parameters
    ----------
    df
        Long-format frame, already feature-engineered and sorted.
    feature_cols
        Columns to stack into each window's channel dimension.
    window, stride
        Window length (timesteps) and hop between consecutive windows.
    group_col
        Windows never cross a change in this column.
    label_col
        If given, the returned ``y`` holds the label of each window's *last*
        timestep; otherwise ``y`` is ``None``.

    Returns
    -------
    X : ndarray, shape (n_windows, window, n_features), float32
    last_index : ndarray, shape (n_windows,)
        The **positional row index into ``df``** of each window's last step
        (use ``df.iloc[last_index]`` to recover metadata / row_id).
    y : ndarray or None, shape (n_windows,)
    """
    # Require a clean 0..n-1 index so group indices are positional.
    df = df.reset_index(drop=True)
    gather = window_gather_index(df, window=window, stride=stride,
                                 group_col=group_col)

    if len(gather) == 0:
        n_feat = len(feature_cols)
        return (np.empty((0, window, n_feat), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int) if label_col else None)

    # float32 up front: the windows are consumed by torch as float32 anyway, so
    # casting the source avoids materialising a float64 copy 8x the row count.
    values = df[feature_cols].to_numpy(dtype=np.float32)
    X = values[gather]
    last_index = gather[:, -1].astype(int)
    y = None
    if label_col:
        y = df[label_col].to_numpy()[last_index].astype(int)
    return X, last_index, y


def scores_to_rows(n_rows: int, last_index: np.ndarray, win_scores: np.ndarray,
                   fill: float = 0.0) -> np.ndarray:
    """Scatter per-window scores back onto a per-row vector of length ``n_rows``.

    Rows that were never the last step of any window (the leading ``window-1``
    rows of each element) receive ``fill``.
    """
    out = np.full(n_rows, fill, dtype=float)
    out[last_index] = win_scores
    return out
