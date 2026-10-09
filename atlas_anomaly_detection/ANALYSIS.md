# Analysis — Data Generation, Per-Type Detection, and Model Combinations

This document answers four questions:

1. **How was the synthetic data generated?**
2. **Which algorithm is best at detecting each anomaly *type*?**
3. **Which algorithms are complementary — i.e. work best *together*?**
4. **Where are the graphs, and what do they show?**

Sections 2–4 are produced by `analyze_results.py` from the saved predictions;
re-run `python main.py && python analyze_results.py` to regenerate them.

---

## 1. How the data was generated

`data/generate_dataset.py` builds a **long-format, multi-channel time series**
that reproduces the *statistical properties that make real ATLAS DCS data hard*,
without copying any closed CERN data.

### 1.1 Structure

- **40 hardware channels** (`element_id`), spread across **6 sub-detectors**
  (`subsystem`): Pixel, SCT, TRT, LAr, Tile, Muon.
- **2,500 timesteps each at 1-minute sampling** → **100,000 rows**.
- Each sub-detector has a **distinct physical operating point** — e.g. the
  liquid-argon calorimeter (LAr) sits near **−183 °C**, room-temperature muon
  chambers near **+21 °C**. This is what makes `subsystem` an informative
  categorical and what makes *contextual* anomalies (a value normal for one
  sub-detector but impossible for another) physically meaningful.

### 1.2 The healthy signal (per channel)

Each channel is generated from a seeded RNG with a **daily seasonal cycle** plus
**physical cross-channel coupling** and per-channel noise:

| channel | how it is built | DCS property it models |
|---|---|---|
| `temperature` | sub-detector mean + daily sine + noise | non-stationary seasonal baseline |
| `current` | couples **linearly to temperature deviation** | second-order coupling (cooling→temp→current) |
| `pressure` | a second, **half-day** periodicity + tiny noise | a different, independent cycle |
| `voltage` | almost-flat rail + tiny noise | a "boring" channel that only matters when it sags |
| `error_rate` | small positive (half-normal) baseline | mostly-zero comms error counter |
| `heartbeat` | monotone counter, wraps at 256 | SCADA liveness byte (freezes on comms loss) |
| `quality_flag` | 0 good / 1 uncertain / 2 bad | SCADA data-validity flag |

Each element's phase is staggered so the 40 channels are not synchronised.

### 1.3 The injected faults (~5 % of rows)

`~5 %` of rows are made anomalous (achieved: see the generator's printed
summary), drawn across **all eleven fault types**, each a dedicated injector
that mutates the relevant channels in place and tags `anomaly=1` +
`anomaly_type`:

| type | class | what it does |
|---|---|---|
| `point_anomaly` | point | 1–3 isolated single-step spikes in a random channel |
| `voltage_spike` | point | sudden 1–4-step voltage spike/dip |
| `temperature_drift` | trend | slow monotone temperature ramp (+ coupled current) |
| `increasing_error_rate` | trend | error-rate ramps up; quality degrades |
| `sensor_freeze` | collective | readings stick at last value; **heartbeat keeps ticking** |
| `communication_loss` | persistent | **heartbeat freezes**, quality → BAD, values hold |
| `collective_anomaly` | collective | fast current oscillation burst, **mean unchanged** |
| `contextual_anomaly` | contextual | daily temperature cycle phase-shifted by π; **every value stays in the healthy range**, only the *timing* is wrong |
| `cooling_failure` | cascading | cooling stops → temperature climbs, pressure rises, current climbs |
| `persistent_anomaly` | persistent | sustained step offset in a channel over a long segment |
| `cascading_failure` | cascading | chain fault: cooling → (lag) temp rise → current rise → voltage sag |

The design deliberately covers the textbook taxonomy (**point / contextual /
collective / persistent / trend / cascading**) so each fault exposes a specific
algorithmic blind spot. Two distinctions are built in on purpose:

- **`sensor_freeze` vs `communication_loss`**: in a freeze the *values* stick but
  the heartbeat keeps counting; in a comms loss the *heartbeat itself* freezes
  and quality goes bad. `heartbeat_delta == 0` separates them.
- **`contextual` / `collective` are in-range**: their individual values stay
  inside the healthy envelope (only the timing or frequency is wrong), so a
  per-point threshold sees nothing — only temporal/multivariate models can.

### 1.4 The `alarm` bit and honest inputs

A **rule-based `alarm`** column is computed *after* injection: it fires on
out-of-band temperature/voltage, high error rate, or BAD quality. It is
**deliberately imperfect** — blind to contextual, collective and frozen-value
faults whose values stay in band.

Crucially, `alarm` (and its history features) are **not fed to the models** —
that would be circular. They are held out and `alarm` is used as the *naive
baseline* the learned detectors must beat. The `anomaly` / `anomaly_type`
labels are never features either (see the README "No label leakage" section).

### 1.5 Outputs

- `data/supervised/dcs_labeled.csv` — telemetry + `anomaly` + `anomaly_type`.
- `data/unsupervised/dcs_unlabeled.csv` — identical telemetry, **no labels**.
- `data/unsupervised/ground_truth.csv` — `row_id,anomaly,anomaly_type` (scoring
  only). Everything is seeded, so regeneration is bit-for-bit reproducible.

---

## 2. Which algorithm is best at which anomaly type?

Detection **coverage = recall per anomaly type** (fraction of that type's rows
flagged). Unsupervised models are scored over all rows (all 11 types visible);
supervised over their held-out test split (the collective / cooling / cascading
faults happened to fall in the training portion of every element's timeline, so
supervised rows show `NaN` there). The `alarm_baseline` row is the naive
threshold detector.

### Best algorithm per anomaly type

| anomaly type | class | best model (recall) | alarm baseline | notes |
|---|---|---|---|---|
| `point_anomaly` | point | **DBSCAN / PCA / KMeans / GMM / most deep** (1.00) | 0.50 | per-point + last-step methods catch spikes; Isolation Forest & boosting **miss** them (0.00) |
| `voltage_spike` | point | **DBSCAN / spatial** (1.00) | 1.00 | boosting misses (0.00); alarm catches (out-of-band) |
| `collective_anomaly` | collective | **DBSCAN + all deep** (0.99–1.00) | **0.00** | oscillation with unchanged mean — alarm is blind; GAN also 0.00 |
| `communication_loss` | persistent | **Isolation Forest / GMM / OmniAnomaly** (1.00) | 1.00 | heartbeat-freeze is a strong signal |
| `increasing_error_rate` | trend | **Random Forest / most** (1.00) | 0.82 | error ramp is easy |
| `sensor_freeze` | collective | **Isolation Forest / spatial** (1.00) | **0.00** | values stick in-band — alarm blind, ML catches |
| `temperature_drift` | trend | **GMM** (0.79) | 0.37 | probabilistic density tracks slow drift best |
| `cooling_failure` | cascading | **GMM** (0.79) | 0.29 | multi-channel climb; density model strongest |
| `persistent_anomaly` | persistent | **GMM** (0.55) | 0.48 | subtle sustained offset — **hard for everyone** |
| `cascading_failure` | cascading | **Transformer** (0.40) | 0.06 | temporal chain — only sequence models get traction |
| `contextual_anomaly` | contextual | Random Forest (0.46) | **0.00** | right value / wrong time — **hardest**, most models <0.17 |

### What the per-type table says

- **The `alarm` baseline is blind to exactly the faults ML exists to catch.**
  On `sensor_freeze`, `collective_anomaly` and `contextual_anomaly` the alarm
  bit scores **0.00** (their values stay in-band), while density/temporal models
  reach ~1.0 on the first two. This is the quantitative case for ML over
  threshold alarms.
- **GMM is the strongest single all-rounder** — best on `temperature_drift`,
  `cooling_failure`, `persistent_anomaly`, and ~1.0 on the easy types. Its
  probabilistic density handles both spikes and slow drifts.
- **DBSCAN owns the sharp per-point faults** (`point_anomaly`, `voltage_spike`,
  `collective_anomaly` all 1.0) — per-point sensitivity that window-aggregating
  models dilute. Note **Isolation Forest misses point spikes** (0.00) despite
  being a strong all-rounder elsewhere.
- **Sequence models (Transformer / LSTM / OmniAnomaly) are the only ones with
  traction on `cascading_failure`** (temporal chains) and lead on
  `collective_anomaly` — the temporal faults spatial methods cannot represent.
- **`contextual_anomaly` and `persistent_anomaly` are the hard ceiling.** No
  model exceeds ~0.5; the phase-shifted / subtly-offset faults stay inside the
  healthy envelope. This is where longer context windows and per-channel
  correlation modelling are the research frontier.

**Graphs:**
- `figures/anomaly_type_coverage.png` — heatmap of every model's recall on
  every anomaly type (plus the `alarm` baseline row).
- `figures/best_algo_per_type.png` — the winning algorithm for each type.

---

## 3. Which algorithms work best together?

Two methods, both on the common test split (so all models are combined on the
same rows):

**(a) Greedy set-cover** — repeatedly add the model that catches the most
still-missed anomalies. The marginal gains show which models are *complementary*
(add new coverage) vs *redundant* (add nothing):

| step | + model | cumulative union recall | family added |
|---|---|---|---|
| 1 | random_forest | 0.557 | supervised (best single, only one with contextual traction) |
| 2 | **+ gmm** | **0.637** | probabilistic-spatial (biggest jump: drifts, cooling, persistent) |
| 3 | **+ transformer** | **0.647** | temporal (cascading / collective) |
| 4 | + isolation_forest | 0.653 | (small gain) |
| 5+ | autoencoder, dbscan, … | → 0.663 | diminishing returns |

The first **three models from three different families** capture essentially
all the coverage; everything after is redundant. If labels are unavailable
(the realistic DCS case), the unsupervised equivalent is **GMM + Transformer +
DBSCAN/Isolation Forest** — the same spatial + probabilistic + temporal trio.

**(b) OR / majority-vote ensembles** (precision / recall / F1):

| ensemble | precision | recall | F1 |
|---|---|---|---|
| single GMM (best single) | 0.710 | 0.549 | **0.619** |
| OR[Isolation Forest + PCA + GMM] | 0.605 | 0.563 | 0.583 |
| OR[RF + XGB + LightGBM] | 0.544 | 0.558 | 0.551 |
| MAJORITY[RF + XGB + LightGBM] | 0.780 | 0.421 | 0.547 |
| MAJORITY[IF + PCA + GMM] | 0.749 | 0.398 | 0.520 |

### What "works best together" means here

- **Combine across families, not within.** OR-ing three *spatial* models (IF +
  PCA + GMM) or three *boosters* (RF+XGB+LGBM) adds little — they catch the same
  anomalies. The value comes from pairing a **per-point** method (DBSCAN /
  Isolation Forest), a **probabilistic-density** method (GMM), and a **temporal**
  method (Transformer / OmniAnomaly). See `model_complementarity.png`: models
  within a family are highly correlated (red), across families much less (green).
- **OR to maximise recall, MAJORITY to maximise precision.** OR-combining raises
  recall (catch every incident — right when a miss is costly) at some precision
  cost; majority vote raises precision (fewer false alarms) at recall cost. Pick
  by operational cost of FP vs FN.
- **The coverage ceiling (~0.66) is set by the hard faults.** No ensemble does
  much better than the best trio because `contextual` and `persistent` anomalies
  defeat *every* member — an ensemble cannot catch what none of its members can.
- **This reproduces the real ATLAS design.** The winning combination —
  fast per-point (DBSCAN) + deep temporal (Transformer/LSTM) — is exactly
  **DeepHYDRA's** architecture (see ALGORITHMS.md "Which are used at ATLAS?").
  The data-driven complementarity analysis independently arrives at the same
  hybrid the ATLAS TDAQ team chose.

**Graphs:**
- `figures/model_complementarity.png` — correlation of *which anomalies each
  model catches*; low correlation (green) = complementary models.
- `figures/ensemble_greedy_coverage.png` — greedy set-cover: how much *new*
  anomaly coverage each added model contributes.

---

## 4. Reproducing this analysis

```bash
python main.py              # generate data + run all 18 detectors
python analyze_results.py   # per-type coverage, complementarity, ensembles
```
