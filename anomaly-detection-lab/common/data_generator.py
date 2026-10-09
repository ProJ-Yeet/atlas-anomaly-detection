"""
Synthetic ATLAS-DCS-like multivariate telemetry generator.

Simulates 5 correlated sensor channels of a fictional subdetector cooling loop
(1-minute sampling, one seasonal "day" = 1440 steps):

    pixel_temp_C      silicon pixel temperature   (daily cycle, coupled to cooling)
    lar_pressure_bar  liquid-argon loop pressure  (half-day cycle)
    lv_voltage_V      low-voltage rail            (flat, tiny noise)
    hv_current_uA     high-voltage rail current   (linearly coupled to temperature)
    cooling_flow_lpm  cooling loop flow rate      (daily cycle)

The TRAIN split is anomaly-free ("healthy detector"). The TEST split contains
five injected fault types that map onto the textbook anomaly taxonomy used in
the DAQ@LHC workshop slides:

    A1  point anomalies      isolated temp / HV spikes
    A2  cascading fault      cooling flow drops -> temp rises -> HV current
                             rises -> LV rail sags   (persistent + collective,
                             mirrors the "chain failure" example in PrintCERN)
    A3  contextual anomaly   temperature daily cycle phase-shifted: every value
                             stays inside the normal range, only the *timing*
                             is wrong
    A4  collective anomaly   fast HV current oscillation burst, mean unchanged
    A5  sensor drift         slow pressure drift (trend anomaly)

Ground-truth binary labels are stored in the `anomaly` column of the test CSV.
"""

import os

import numpy as np
import pandas as pd

CHANNELS = ["pixel_temp_C", "lar_pressure_bar", "lv_voltage_V",
            "hv_current_uA", "cooling_flow_lpm"]

# (name, start, end_exclusive) of every injected fault in the TEST split
FAULTS = [
    ("A1_point_spike_temp", 300, 302),
    ("A2_cascade", 800, 1100),
    ("A1_point_dip_temp", 1500, 1501),
    ("A3_contextual_phase", 1900, 2100),
    ("A4_collective_hv_osc", 2600, 2750),
    ("A1_point_spike_hv", 3200, 3202),
    ("A5_pressure_drift", 3400, 3900),
]
DAY = 1440  # steps per seasonal cycle (minutes per day)
N_TRAIN = 8000
N_TEST = 4000
SEED = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def _base_signals(n, rng, t0=0):
    """Healthy correlated telemetry."""
    t = np.arange(t0, t0 + n)
    cooling = 12.0 + 0.30 * np.sin(2 * np.pi * t / DAY) + rng.normal(0, 0.05, n)
    temp = (-10.0 + 0.80 * np.sin(2 * np.pi * t / DAY - 0.5)
            - 0.50 * (cooling - 12.0) + rng.normal(0, 0.06, n))
    pressure = (1.30 + 0.020 * np.sin(2 * np.pi * t / (DAY / 2) + 1.0)
                + rng.normal(0, 0.002, n))
    lv = 3.30 + rng.normal(0, 0.004, n)
    hv = 40.0 + 2.0 * (temp + 10.0) + rng.normal(0, 0.40, n)
    return np.stack([temp, pressure, lv, hv, cooling], axis=1), t


def _inject_anomalies(X, t, rng):
    """Inject the five fault types into a test batch. Returns (X, labels)."""
    n = X.shape[0]
    y = np.zeros(n, dtype=int)
    temp, pres, lv, hv, cool = (X[:, i] for i in range(5))

    # A1 -- point anomalies: temp / HV spikes
    for idx, ch, delta in [((300, 302), temp, +3.0),
                           ((1500, 1501), temp, -2.5),
                           ((3200, 3202), hv, +15.0)]:
        s, e = idx
        ch[s:e] += delta
        y[s:e] = 1

    # A2 -- cascading fault (persistent): valve degrades 800..1100
    s, e = 800, 1100
    m = e - s
    ramp = np.linspace(0, 1, m) ** 1.5
    cool[s:e] -= 3.0 * ramp                       # mechanical: flow drops
    lag = 30                                       # thermal inertia
    temp[s + lag:e] += 2.5 * ramp[:m - lag]        # environmental: temp rises
    hv[s + lag:e] += 5.0 * ramp[:m - lag]          # power: HV current rises
    lv[e - 100:e] -= np.linspace(0, 0.05, 100)     # power: LV rail sags late
    y[s:e] = 1

    # A3 -- contextual anomaly: daily temp cycle phase-shifted by pi,
    # every individual value stays within the healthy global envelope
    s, e = 1900, 2100
    seg = np.arange(s, e)
    healthy_season = 0.80 * np.sin(2 * np.pi * t[seg] / DAY - 0.5)
    shifted_season = 0.80 * np.sin(2 * np.pi * t[seg] / DAY - 0.5 + np.pi)
    temp[s:e] += shifted_season - healthy_season
    y[s:e] = 1

    # A4 -- collective anomaly: fast HV oscillation burst, mean unchanged
    s, e = 2600, 2750
    hv[s:e] += 3.0 * np.sin(2 * np.pi * np.arange(e - s) / 8.0)
    y[s:e] = 1

    # A5 -- trend anomaly: slow pressure sensor drift
    s, e = 3400, 3900
    pres[s:e] += np.linspace(0, 0.06, e - s)
    y[s:e] = 1

    return X, y


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    X_train, _ = _base_signals(N_TRAIN, rng, t0=0)
    X_test, t_test = _base_signals(N_TEST, rng, t0=N_TRAIN)
    X_test, y_test = _inject_anomalies(X_test, t_test, rng)

    train = pd.DataFrame(X_train, columns=CHANNELS)
    test = pd.DataFrame(X_test, columns=CHANNELS)
    test["anomaly"] = y_test
    return train, test


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    train, test = generate()
    train.to_csv(os.path.join(DATA_DIR, "dcs_train.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "dcs_test.csv"), index=False)
    print(f"wrote data/dcs_train.csv {train.shape}, data/dcs_test.csv {test.shape} "
          f"({test['anomaly'].mean():.1%} anomalous)")


if __name__ == "__main__":
    main()
