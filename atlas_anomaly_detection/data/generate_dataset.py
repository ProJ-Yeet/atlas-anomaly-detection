"""Synthetic ATLAS DCS ``EVENTHISTORY``-style telemetry generator.

The real ATLAS Detector Control System (DCS) archive is closed (the 68 TB
``EVENTHISTORY`` Oracle store is only reachable with CERN credentials through
the DDT API), so this module fabricates a *statistically faithful* stand-in for
building and validating an anomaly-detection pipeline offline.

What it models
--------------
* **Many hardware channels.** ``N_ELEMENTS`` distinct ``element_id`` sensors,
  spread over six real ATLAS sub-detectors (Pixel, SCT, TRT, LAr, Tile, Muon).
  Each sub-detector has its own physical operating point (a liquid-argon
  calorimeter sits near -183 C, a room-temperature muon chamber near +20 C),
  which is what makes ``subsystem`` an informative categorical feature and what
  makes *contextual* anomalies (a value that is normal for one sub-detector but
  impossible for another) meaningful.
* **Correlated multivariate signals.** temperature / pressure / voltage /
  current / error_rate are coupled (cooling drives temperature, temperature
  drives current, ...) with a daily seasonal cycle, exactly the structure that
  defeats naive per-channel threshold alarms.
* **SCADA bookkeeping columns.** ``heartbeat`` (a liveness counter),
  ``quality_flag`` (0 good / 1 uncertain / 2 bad) and ``alarm`` (the detector's
  *own* rule-based alarm bit). ``alarm`` is deliberately imperfect: it is a
  fixed-threshold rule, so it fires on out-of-range faults but is blind to
  contextual / collective / frozen-value faults -- which is the whole reason a
  learned detector is worth building.

Injected fault taxonomy (all eleven types the project brief asks for)
---------------------------------------------------------------------
``temperature_drift``, ``voltage_spike``, ``communication_loss``,
``sensor_freeze``, ``increasing_error_rate``, ``cooling_failure``,
``cascading_failure``, ``contextual_anomaly``, ``point_anomaly``,
``collective_anomaly``, ``persistent_anomaly``.

Outputs (written next to this file)
-----------------------------------
* ``supervised/dcs_labeled.csv``    -- every column **plus** the ground-truth
  ``anomaly`` (0/1) and ``anomaly_type`` columns. Feeds the supervised models.
* ``unsupervised/dcs_unlabeled.csv`` -- the identical telemetry **without** any
  label column. This is the honest input for the unsupervised models: they
  never see a label at train time.
* ``unsupervised/ground_truth.csv``  -- ``row_id,anomaly,anomaly_type`` only,
  used *exclusively* for scoring the unsupervised models after the fact.

Everything is seeded, so re-running reproduces the dataset bit-for-bit.

Run standalone::

    python data/generate_dataset.py --n-elements 40 --steps 2500 --seed 42
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
SUP_DIR = os.path.join(HERE, "supervised")
UNSUP_DIR = os.path.join(HERE, "unsupervised")

DAY = 1440  # minutes per day -> one seasonal cycle at 1-minute sampling
TARGET_ANOMALY_RATE = 0.05  # ~5% of rows carry a ground-truth anomaly

# The eleven fault types the brief requires. The weights control roughly how
# often each is drawn; they need not sum to one (they are normalised).
ANOMALY_TYPES: List[str] = [
    "point_anomaly",
    "voltage_spike",
    "temperature_drift",
    "increasing_error_rate",
    "sensor_freeze",
    "communication_loss",
    "collective_anomaly",
    "contextual_anomaly",
    "cooling_failure",
    "persistent_anomaly",
    "cascading_failure",
]
ANOMALY_WEIGHTS = np.array(
    [0.10, 0.10, 0.12, 0.10, 0.09, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06]
)


@dataclass
class SubsystemSpec:
    """Physical operating point of one ATLAS sub-detector.

    Attributes are the mean level and daily-cycle amplitude of each channel,
    plus the fixed-threshold band the detector's own ``alarm`` rule uses.
    """

    name: str
    temp_mean: float          # C
    temp_amp: float           # C, daily swing
    pressure_mean: float      # bar
    voltage_mean: float       # V
    current_mean: float       # uA
    # Rule-based alarm bands (absolute deviation from mean that trips ``alarm``)
    temp_band: float
    voltage_band: float
    error_band: float = 0.15


# Six sub-detectors with physically distinct operating points.
SUBSYSTEMS: List[SubsystemSpec] = [
    SubsystemSpec("Pixel", temp_mean=-10.0, temp_amp=0.8, pressure_mean=1.05,
                  voltage_mean=3.30, current_mean=42.0, temp_band=3.0,
                  voltage_band=0.06),
    SubsystemSpec("SCT", temp_mean=-7.0, temp_amp=0.7, pressure_mean=1.10,
                  voltage_mean=3.50, current_mean=38.0, temp_band=3.0,
                  voltage_band=0.06),
    SubsystemSpec("TRT", temp_mean=20.0, temp_amp=1.2, pressure_mean=1.00,
                  voltage_mean=5.00, current_mean=25.0, temp_band=4.0,
                  voltage_band=0.10),
    SubsystemSpec("LAr", temp_mean=-183.0, temp_amp=0.4, pressure_mean=1.25,
                  voltage_mean=2.00, current_mean=60.0, temp_band=2.0,
                  voltage_band=0.05),
    SubsystemSpec("Tile", temp_mean=22.0, temp_amp=1.5, pressure_mean=1.00,
                  voltage_mean=5.00, current_mean=20.0, temp_band=4.0,
                  voltage_band=0.10),
    SubsystemSpec("Muon", temp_mean=21.0, temp_amp=1.3, pressure_mean=1.00,
                  voltage_mean=4.50, current_mean=18.0, temp_band=4.0,
                  voltage_band=0.10),
]


@dataclass
class ElementSeries:
    """Mutable per-element channel arrays during generation."""

    n: int
    temperature: np.ndarray
    pressure: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    error_rate: np.ndarray
    heartbeat: np.ndarray
    quality_flag: np.ndarray
    label: np.ndarray = field(init=False)
    atype: List[str] = field(init=False)

    def __post_init__(self) -> None:
        self.label = np.zeros(self.n, dtype=int)
        self.atype = ["normal"] * self.n

    def mark(self, s: int, e: int, name: str) -> None:
        """Tag steps ``[s:e)`` as anomalous of type ``name``."""
        self.label[s:e] = 1
        for i in range(s, e):
            self.atype[i] = name


# --------------------------------------------------------------------------- #
# Healthy signal
# --------------------------------------------------------------------------- #

def _healthy_series(n: int, spec: SubsystemSpec, rng: np.random.Generator,
                    t0: int) -> ElementSeries:
    """Generate one anomaly-free element time series.

    The channels are physically coupled:  a daily cooling cycle drives
    ``temperature``; ``current`` follows ``temperature`` linearly; ``pressure``
    has a second, faster periodicity; ``voltage`` is an almost-flat rail (it
    only matters when it sags).  ``error_rate`` is a small positive baseline.
    """
    t = np.arange(t0, t0 + n)
    daily = np.sin(2.0 * np.pi * t / DAY)

    cooling = spec.temp_amp * daily + rng.normal(0.0, 0.05, n)
    temperature = spec.temp_mean + cooling + rng.normal(0.0, 0.04, n)
    pressure = (spec.pressure_mean
                + 0.02 * np.sin(2.0 * np.pi * t / (DAY / 2.0) + 1.0)
                + rng.normal(0.0, 0.003, n))
    voltage = spec.voltage_mean + rng.normal(0.0, 0.004, n)
    # current couples to temperature deviation from the operating point.
    current = (spec.current_mean
               + 1.5 * (temperature - spec.temp_mean)
               + rng.normal(0.0, 0.30, n))
    # error_rate: small positive baseline (half-normal), mostly ~0.
    error_rate = np.abs(rng.normal(0.0, 0.01, n))
    # heartbeat: monotone liveness counter, wraps at 256 (like a SCADA byte).
    heartbeat = ((np.arange(n) + int(rng.integers(0, 256))) % 256).astype(float)
    quality_flag = np.zeros(n, dtype=int)  # 0 = GOOD

    return ElementSeries(n=n, temperature=temperature, pressure=pressure,
                         voltage=voltage, current=current,
                         error_rate=error_rate, heartbeat=heartbeat,
                         quality_flag=quality_flag)


# --------------------------------------------------------------------------- #
# Fault injectors -- one per taxonomy entry
# --------------------------------------------------------------------------- #

def _inject(es: ElementSeries, spec: SubsystemSpec, kind: str,
            rng: np.random.Generator) -> int:
    """Inject one fault episode of ``kind`` into ``es``; return #rows tagged.

    Each branch mutates the relevant channels in place and calls
    :meth:`ElementSeries.mark`.  Segment lengths are randomised so no two
    episodes are identical.
    """
    n = es.n

    def seg(min_len: int, max_len: int) -> Tuple[int, int]:
        length = int(rng.integers(min_len, max_len + 1))
        s = int(rng.integers(30, max(31, n - length - 30)))
        return s, s + length

    if kind == "point_anomaly":
        # 1-3 isolated spikes in a randomly chosen channel.
        count = int(rng.integers(1, 4))
        for _ in range(count):
            s = int(rng.integers(30, n - 2))
            ch = rng.choice(["temperature", "current", "pressure"])
            arr = getattr(es, ch)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            arr[s] += sign * (6.0 if ch == "temperature"
                              else 20.0 if ch == "current" else 0.3)
            es.mark(s, s + 1, "point_anomaly")
        return int(es.label.sum() > 0)

    if kind == "voltage_spike":
        # Sudden short voltage spike/dip (point-like, 1-4 steps).
        s, e = seg(1, 4)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        es.voltage[s:e] += sign * (0.4 + 0.3 * rng.random())
        es.mark(s, e, "voltage_spike")
        return e - s

    if kind == "temperature_drift":
        # Slow monotone temperature drift (trend anomaly).
        s, e = seg(120, 260)
        drift = 4.0 + 3.0 * rng.random()
        es.temperature[s:e] += np.linspace(0.0, drift, e - s)
        es.current[s:e] += np.linspace(0.0, 1.5 * drift, e - s)
        es.mark(s, e, "temperature_drift")
        return e - s

    if kind == "increasing_error_rate":
        # Error rate ramps up (degrading link / firmware fault).
        s, e = seg(100, 220)
        es.error_rate[s:e] += np.linspace(0.0, 0.6 + 0.4 * rng.random(), e - s)
        es.quality_flag[s:e] = np.where(
            es.error_rate[s:e] > 0.3, 1, es.quality_flag[s:e])
        es.mark(s, e, "increasing_error_rate")
        return e - s

    if kind == "sensor_freeze":
        # Readings stick at the last value; heartbeat KEEPS incrementing
        # (this is what distinguishes a frozen sensor from a comms loss).
        s, e = seg(60, 160)
        for ch in ("temperature", "pressure", "voltage", "current"):
            arr = getattr(es, ch)
            arr[s:e] = arr[s - 1]
        es.quality_flag[s:e] = 1
        es.mark(s, e, "sensor_freeze")
        return e - s

    if kind == "communication_loss":
        # Heartbeat FREEZES, quality goes BAD, readings hold last value.
        s, e = seg(50, 140)
        es.heartbeat[s:e] = es.heartbeat[s - 1]
        es.quality_flag[s:e] = 2
        for ch in ("temperature", "pressure", "voltage", "current"):
            arr = getattr(es, ch)
            arr[s:e] = arr[s - 1]
        es.error_rate[s:e] += 0.2
        es.mark(s, e, "communication_loss")
        return e - s

    if kind == "collective_anomaly":
        # Fast current oscillation burst; marginal mean is unchanged, so a
        # per-point threshold sees nothing -- only the *pattern* is wrong.
        s, e = seg(80, 160)
        amp = 8.0 + 4.0 * rng.random()
        es.current[s:e] += amp * np.sin(2.0 * np.pi * np.arange(e - s) / 6.0)
        es.mark(s, e, "collective_anomaly")
        return e - s

    if kind == "contextual_anomaly":
        # Phase-shift the daily temperature cycle by pi: every individual value
        # stays inside the healthy global envelope, only the *timing* is wrong.
        s, e = seg(120, 220)
        t = np.arange(s, e)
        healthy = spec.temp_amp * np.sin(2.0 * np.pi * (t) / DAY)
        shifted = spec.temp_amp * np.sin(2.0 * np.pi * (t) / DAY + np.pi)
        es.temperature[s:e] += shifted - healthy
        es.mark(s, e, "contextual_anomaly")
        return e - s

    if kind == "cooling_failure":
        # Cooling stops -> temperature climbs, pressure rises, current climbs.
        s, e = seg(120, 240)
        ramp = np.linspace(0.0, 1.0, e - s) ** 1.3
        es.temperature[s:e] += (6.0 + 3.0 * rng.random()) * ramp
        es.pressure[s:e] += 0.15 * ramp
        es.current[s:e] += 12.0 * ramp
        es.quality_flag[s:e] = np.where(ramp > 0.6, 1, es.quality_flag[s:e])
        es.mark(s, e, "cooling_failure")
        return e - s

    if kind == "persistent_anomaly":
        # Sustained step offset in a channel for a long segment.
        s, e = seg(180, 320)
        ch = rng.choice(["voltage", "current", "pressure"])
        arr = getattr(es, ch)
        offset = {"voltage": 0.15, "current": 8.0, "pressure": 0.08}[ch]
        arr[s:e] += (1.0 if rng.random() < 0.5 else -1.0) * offset
        es.mark(s, e, "persistent_anomaly")
        return e - s

    if kind == "cascading_failure":
        # The PrintCERN "chain failure": cooling drops -> (lag) temp rises ->
        # current rises -> voltage rail sags late. Persistent + collective.
        s, e = seg(160, 300)
        m = e - s
        ramp = np.linspace(0.0, 1.0, m) ** 1.5
        lag = 25
        es.temperature[s:e] += 5.0 * ramp
        if m > lag:
            es.current[s + lag:e] += 10.0 * ramp[:m - lag]
        es.voltage[max(s, e - 90):e] -= np.linspace(
            0.0, 0.12, min(90, e - max(s, e - 90)))
        es.quality_flag[s:e] = np.where(ramp > 0.5, 1, es.quality_flag[s:e])
        es.mark(s, e, "cascading_failure")
        return e - s

    raise ValueError(f"unknown anomaly kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Alarm rule (deliberately imperfect)
# --------------------------------------------------------------------------- #

def _rule_based_alarm(es: ElementSeries, spec: SubsystemSpec) -> np.ndarray:
    """The detector's *own* fixed-threshold alarm bit.

    Fires on out-of-band temperature/voltage, high error rate, or BAD quality.
    It is intentionally blind to contextual, collective and frozen-value faults
    (their values stay in band) -- illustrating why a learned model adds value.
    """
    temp_dev = np.abs(es.temperature - spec.temp_mean) > spec.temp_band
    volt_dev = np.abs(es.voltage - spec.voltage_mean) > spec.voltage_band
    err_hi = es.error_rate > spec.error_band
    bad_q = es.quality_flag >= 2
    return (temp_dev | volt_dev | err_hi | bad_q).astype(int)


# --------------------------------------------------------------------------- #
# Top-level generation
# --------------------------------------------------------------------------- #

def generate(n_elements: int = 40, steps: int = 2500, seed: int = 42
             ) -> pd.DataFrame:
    """Build the full long-format DataFrame (~``n_elements * steps`` rows)."""
    rng = np.random.default_rng(seed)
    weights = ANOMALY_WEIGHTS / ANOMALY_WEIGHTS.sum()

    start_time = pd.Timestamp("2026-01-01 00:00:00")
    frames: List[pd.DataFrame] = []
    tagged = 0
    total = n_elements * steps

    for el in range(n_elements):
        spec = SUBSYSTEMS[el % len(SUBSYSTEMS)]
        element_id = f"{spec.name}_{el // len(SUBSYSTEMS):02d}"
        # Stagger each element's phase so they are not perfectly synchronised.
        t0 = int(rng.integers(0, DAY))
        es = _healthy_series(steps, spec, rng, t0)

        # Inject anomalies until we approach the ~5% global budget. Most
        # elements receive one to three episodes; the budget cap (a soft
        # ceiling slightly above target) prevents runaway over-injection.
        if rng.random() < 0.88 and tagged < 1.05 * TARGET_ANOMALY_RATE * total:
            n_episodes = int(rng.integers(1, 4))
            for _ in range(n_episodes):
                kind = str(rng.choice(ANOMALY_TYPES, p=weights))
                tagged += _inject(es, spec, kind, rng)

        alarm = _rule_based_alarm(es, spec)
        timestamps = start_time + pd.to_timedelta(np.arange(steps), unit="m")

        frames.append(pd.DataFrame({
            "timestamp": timestamps,
            "element_id": element_id,
            "subsystem": spec.name,
            "temperature": np.round(es.temperature, 4),
            "pressure": np.round(es.pressure, 5),
            "voltage": np.round(es.voltage, 5),
            "current": np.round(es.current, 4),
            "error_rate": np.round(es.error_rate, 5),
            "heartbeat": es.heartbeat.astype(int),
            "quality_flag": es.quality_flag.astype(int),
            "alarm": alarm,
            "anomaly": es.label,
            "anomaly_type": es.atype,
        }))

    df = pd.concat(frames, ignore_index=True)
    # Stable global row id so the unlabeled frame and the ground-truth file
    # can be joined back together for scoring.
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def write_datasets(df: pd.DataFrame) -> None:
    """Write the labeled, unlabeled and ground-truth CSVs to disk."""
    os.makedirs(SUP_DIR, exist_ok=True)
    os.makedirs(UNSUP_DIR, exist_ok=True)

    # 1) Supervised: keep the label + type.
    df.to_csv(os.path.join(SUP_DIR, "dcs_labeled.csv"), index=False)

    # 2) Unsupervised input: identical telemetry, NO label columns.
    unlabeled = df.drop(columns=["anomaly", "anomaly_type"])
    unlabeled.to_csv(os.path.join(UNSUP_DIR, "dcs_unlabeled.csv"), index=False)

    # 3) Ground truth for scoring the unsupervised models (only).
    df[["row_id", "anomaly", "anomaly_type"]].to_csv(
        os.path.join(UNSUP_DIR, "ground_truth.csv"), index=False)


def _summary(df: pd.DataFrame) -> str:
    rate = df["anomaly"].mean()
    by_type = (df[df["anomaly"] == 1]["anomaly_type"]
               .value_counts().to_dict())
    lines = [
        f"rows              : {len(df):,}",
        f"elements          : {df['element_id'].nunique()}",
        f"subsystems        : {df['subsystem'].nunique()}",
        f"anomaly rate      : {rate:.2%}",
        f"alarm rate        : {df['alarm'].mean():.2%}",
        f"anomaly types     :",
    ]
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k:<24s} {v:>6,}  ({v / len(df):.2%})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-elements", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.n_elements, args.steps, args.seed)
    write_datasets(df)
    print(_summary(df))
    print(f"\nwrote:\n  {os.path.join(SUP_DIR, 'dcs_labeled.csv')}"
          f"\n  {os.path.join(UNSUP_DIR, 'dcs_unlabeled.csv')}"
          f"\n  {os.path.join(UNSUP_DIR, 'ground_truth.csv')}")


if __name__ == "__main__":
    main()
