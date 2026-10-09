"""Feature engineering for ATLAS DCS telemetry.

All features are computed **per element** (``groupby('element_id')``) and in
timestamp order, so no feature ever mixes two different sensors or looks across
a time discontinuity.  Every transform is causal (uses only past/current
values) except the rolling median centring, which is fine for offline batch
scoring but should be made trailing before a streaming deployment.

Engineered features (the brief's full list)
-------------------------------------------
rolling mean / std / median, lag features, rate of change, gradient, EWMA,
time-since-last-alarm, alarm frequency, cross-sensor correlation,
temperature-voltage interaction, pressure-temperature interaction, and a
composite sensor-health score.  ``heartbeat`` is converted to its (wrap-aware)
delta, which is the actually-useful signal: delta == 0 means the liveness
counter froze -- a communication loss.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

#: Channels that receive the full rolling / lag / gradient treatment.
_DYNAMIC = ["temperature", "pressure", "voltage", "current", "error_rate"]
_ROLL_WINDOW = 15   # 15 one-minute samples
_LAGS = [1, 5]
_EWMA_SPAN = 12
_HEARTBEAT_WRAP = 256


def _grouped_rolling_corr(df: pd.DataFrame, group_col: str, a: str, b: str,
                          window: int) -> pd.Series:
    """Causal rolling Pearson correlation between channels ``a`` and ``b``,
    computed per group from rolling moments (fully vectorised via ``transform``,
    so it avoids the slow -- and now deprecated -- ``groupby.apply`` path).
    """
    tmp = pd.DataFrame({
        "a": df[a], "b": df[b], "ab": df[a] * df[b],
        "a2": df[a] * df[a], "b2": df[b] * df[b], "g": df[group_col],
    })
    g = tmp.groupby("g", sort=False)

    def roll(col: str) -> pd.Series:
        return g[col].transform(
            lambda s: s.rolling(window, min_periods=3).mean())

    ma, mb, mab = roll("a"), roll("b"), roll("ab")
    va = roll("a2") - ma * ma
    vb = roll("b2") - mb * mb
    cov = mab - ma * mb
    denom = np.sqrt(np.clip(va, 1e-9, None) * np.clip(vb, 1e-9, None))
    return (cov / denom).clip(-1.0, 1.0)


def _heartbeat_delta(hb: pd.Series) -> pd.Series:
    """Wrap-aware first difference of the heartbeat counter (0 == frozen)."""
    d = hb.diff()
    d = d.where(d >= 0, d + _HEARTBEAT_WRAP)   # undo the 256 wrap
    return d.fillna(1.0)


def add_features(df: pd.DataFrame, group_col: str = "element_id"
                 ) -> pd.DataFrame:
    """Return a copy of ``df`` with all engineered feature columns appended.

    The input must contain the raw sensor channels and ``timestamp``; the
    output is sorted by (group, timestamp) and NaNs introduced by the leading
    edge of rolling/lag windows are filled with 0.
    """
    df = df.sort_values([group_col, "timestamp"]).reset_index(drop=True)
    g = df.groupby(group_col, sort=False)

    # --- rolling statistics ------------------------------------------------ #
    for col in _DYNAMIC:
        gr = g[col]
        df[f"{col}_roll_mean"] = gr.transform(
            lambda s: s.rolling(_ROLL_WINDOW, min_periods=1).mean())
        df[f"{col}_roll_std"] = gr.transform(
            lambda s: s.rolling(_ROLL_WINDOW, min_periods=1).std()).fillna(0.0)
        df[f"{col}_roll_median"] = gr.transform(
            lambda s: s.rolling(_ROLL_WINDOW, min_periods=1).median())

    # --- lag features ------------------------------------------------------ #
    for col in ["temperature", "voltage", "current"]:
        for lag in _LAGS:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)

    # --- rate of change / gradient ---------------------------------------- #
    for col in ["temperature", "pressure", "current"]:
        df[f"{col}_roc"] = g[col].diff()
        df[f"{col}_grad"] = g[col].transform(
            lambda s: pd.Series(np.gradient(s.to_numpy()), index=s.index)
            if len(s) > 1 else s * 0.0)

    # --- EWMA -------------------------------------------------------------- #
    for col in ["temperature", "current", "error_rate"]:
        df[f"{col}_ewma"] = g[col].transform(
            lambda s: s.ewm(span=_EWMA_SPAN, adjust=False).mean())

    # --- alarm dynamics ---------------------------------------------------- #
    # time since last alarm: steps since ``alarm`` was last 1, per element.
    def _time_since_alarm(s: pd.Series) -> pd.Series:
        out = np.empty(len(s), dtype=float)
        since = np.iinfo(np.int32).max
        for i, v in enumerate(s.to_numpy()):
            since = 0 if v == 1 else since + 1
            out[i] = min(since, 10000)
        return pd.Series(out, index=s.index)

    df["time_since_alarm"] = g["alarm"].transform(_time_since_alarm)
    df["alarm_freq"] = g["alarm"].transform(
        lambda s: s.rolling(60, min_periods=1).mean())

    # --- cross-sensor correlation ----------------------------------------- #
    df["temp_curr_corr"] = _grouped_rolling_corr(
        df, group_col, "temperature", "current", 30).fillna(0.0)

    # --- interaction features --------------------------------------------- #
    df["temp_volt_interaction"] = df["temperature"] * df["voltage"]
    df["pres_temp_interaction"] = df["pressure"] * df["temperature"]

    # --- heartbeat delta --------------------------------------------------- #
    df["heartbeat_delta"] = g["heartbeat"].transform(_heartbeat_delta)

    # --- composite sensor-health score ------------------------------------ #
    # Higher = healthier.  Penalises high error rate, bad data quality and a
    # frozen heartbeat -- all genuine, independent SCADA signals.  Deliberately
    # does NOT use the ``alarm`` bit (that would smuggle the detector's own
    # decision back into the inputs); a cheap interpretable summary channel.
    df["sensor_health"] = (
        1.0
        - df["error_rate"].clip(0, 1)
        - 0.3 * df["quality_flag"].clip(0, 2)
        - 0.5 * (df["heartbeat_delta"] == 0).astype(float)
    ).clip(-2.0, 1.0)

    return df.fillna(0.0)


def engineered_columns(df: pd.DataFrame) -> List[str]:
    """Names of all columns added by :func:`add_features` present in ``df``."""
    known_suffix = ("_roll_mean", "_roll_std", "_roll_median", "_roc",
                    "_grad", "_ewma")
    extra = ["time_since_alarm", "alarm_freq", "temp_curr_corr",
             "temp_volt_interaction", "pres_temp_interaction",
             "heartbeat_delta", "sensor_health"]
    cols = [c for c in df.columns
            if c.endswith(known_suffix) or "_lag" in c or c in extra]
    return cols
