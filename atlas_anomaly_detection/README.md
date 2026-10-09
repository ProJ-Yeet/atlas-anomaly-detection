# ATLAS DCS Anomaly Detection — Research Prototype

A runnable prototype that mimics the **ATLAS Detector Control System (DCS)**
anomaly-detection pipeline. Because the real CERN DCS `EVENTHISTORY` archive is
closed (68 TB behind CERN credentials and the DDT API), this project generates
**realistic synthetic DCS telemetry** with known ground truth and implements
**17 anomaly-detection algorithms** on it — 11 unsupervised, 6 supervised —
plus a full evaluation, feature-engineering and comparison stack.

It is the single-machine sibling of the distributed 14-algorithm Spark/PyTorch
ensemble in
[`CERN-Big-Data-Analysis/distributedAlgorithm14`](https://github.com/snigdhkarki/CERN-Big-Data-Analysis/tree/main/distributedAlgorithm14):
this repo is where you *understand and validate* each algorithm; that repo is
where they run distributed. See **Deployment on Apache Spark** below.

---

## 1. The synthetic dataset

`data/generate_dataset.py` fabricates ~**100,000 records** across **40 hardware
channels** (`element_id`) spread over **6 ATLAS sub-detectors** (`subsystem`:
Pixel, SCT, TRT, LAr, Tile, Muon), each with its own physical operating point
and a daily seasonal cycle. Channels are physically coupled (cooling → temperature
→ current), which is what defeats naive per-channel threshold alarms.

**Schema (DCS `EVENTHISTORY`-like):**

| column | meaning |
|---|---|
| `timestamp` | 1-minute sampling |
| `element_id` | hardware channel id (e.g. `Pixel_03`) |
| `subsystem` | sub-detector |
| `temperature`, `pressure`, `voltage`, `current`, `error_rate` | sensor channels |
| `heartbeat` | liveness counter (freezes on comms loss) |
| `quality_flag` | 0 good / 1 uncertain / 2 bad |
| `alarm` | the detector's **own** rule-based alarm bit (deliberately imperfect) |
| `anomaly` | ground-truth label (labeled dataset only) |
| `anomaly_type` | which fault (labeled dataset only) |

**~5 % of rows are anomalous**, spanning all eleven requested fault types:
`temperature_drift`, `voltage_spike`, `communication_loss`, `sensor_freeze`,
`increasing_error_rate`, `cooling_failure`, `cascading_failure`,
`contextual_anomaly`, `point_anomaly`, `collective_anomaly`,
`persistent_anomaly`.

The `alarm` column is a fixed-threshold rule: it fires on out-of-band
temperature/voltage but is **blind** to contextual, collective and frozen-value
faults — illustrating exactly why a learned detector adds value.

### Two dataset folders (supervised vs unsupervised)

```
data/
  supervised/   dcs_labeled.csv       full telemetry + anomaly + anomaly_type
  unsupervised/ dcs_unlabeled.csv     identical telemetry, NO label columns
                ground_truth.csv      row_id,anomaly,anomaly_type (scoring only)
```

The unsupervised models train on `dcs_unlabeled.csv` and **never see a label**;
`ground_truth.csv` is joined back by `row_id` *only* to compute metrics. This
keeps the unsupervised evaluation honest while still measurable.

### No label leakage — what the models are actually fed

Real DCS telemetry carries **no "this row is anomalous" marker**, so neither do
the model inputs here:

- **`anomaly` / `anomaly_type` are never features.** The unlabeled CSV does not
  contain them, and `utils/preprocessing.feature_columns()` explicitly drops
  them. For unsupervised models the label lives in a separate `y` array used
  *only* by `compute_metrics()` after scoring — `model.fit()` never sees it.
  (Verify: `python -c "from utils.pipeline import prepare_unsupervised as f;
  d=f(); print('anomaly' in d.feature_cols)"` → `False`.)
- **The `alarm` bit is held out too.** It is the detector's *own* threshold-alarm
  decision; feeding it into another anomaly detector would be circular and
  unrealistic. So `alarm` and its derived features (`time_since_alarm`,
  `alarm_freq`) are excluded from every model's inputs, and `alarm` is used
  instead as the **naive baseline to beat** (see `analyze_results.py`). The
  `sensor_health` feature is likewise computed without the alarm bit.
- **`quality_flag` is kept** — it is a genuine, independent SCADA *data-validity*
  signal (is the sensor communicating correctly?), not an anomaly verdict, and
  real DCS detectors legitimately use it.

Supervised models, by definition, use the `anomaly` label as their training
target — that is inherent to supervised learning, and realistic in DCS via
operator-confirmed incidents or synthetic fault-injection campaigns. They are
still evaluated on a held-out temporal split they were not trained on.

---

## 2. Project structure

```
atlas_anomaly_detection/
├── data/
│   ├── generate_dataset.py          synthetic DCS generator (+ 5% anomalies)
│   ├── supervised/                  labeled dataset
│   └── unsupervised/                unlabeled dataset + ground truth
├── algorithms/
│   ├── unsupervised/                isolation_forest, dbscan, pca, kmeans, gmm,
│   │                                autoencoder, lstm_unsupervised, transformer,
│   │                                vae, usad, omnianomaly, gan
│   └── supervised/                  random_forest, xgboost, lightgbm, catboost,
│                                    logistic_regression, svm
├── evaluation/                      metrics, roc_curve, precision_recall,
│                                    confusion_matrix
├── utils/                           preprocessing, feature_engineering,
│                                    windowing, visualization, pipeline,
│                                    torch_utils
├── figures/                         per-model figures + predictions + metrics
├── main.py                          orchestrator (generate → run all → compare)
├── compare_models.py                cross-model comparison table + chart
├── analyze_results.py               per-anomaly-type coverage + ensemble analysis
├── requirements.txt
├── README.md
├── ALGORITHMS.md                    what each algorithm does & why it fits DCS
├── ANALYSIS.md                      data generation, per-type winners, best combos
└── GRAPHS_GUIDE.md                  how to read every figure in figures/
```

Every algorithm script is **independently executable**, loads data, preprocesses,
engineers features, trains, predicts, scores, saves predictions
(`figures/<name>_predictions.csv`), writes metrics
(`figures/<name>_metrics.json`) and renders a **9-panel diagnostic figure**
(time series + anomaly overlay, score trace, score histogram, ROC, PR,
confusion matrix, two PCA scatters, and feature importance for supervised
models).

---

## 3. Feature engineering

`utils/feature_engineering.py` computes, **per element in time order**:
rolling mean/std/median, lag features, rate-of-change, gradient, EWMA,
time-since-last-alarm, alarm frequency, cross-sensor rolling correlation,
temperature–voltage and pressure–temperature interactions, a wrap-aware
`heartbeat_delta` (0 = frozen), and a composite `sensor_health` score.

---

## 4. Quickstart

```bash
pip install -r requirements.txt

# 1) generate the dataset (writes the three CSVs above)
python data/generate_dataset.py

# 2a) run one detector standalone (each is independently runnable)
python algorithms/unsupervised/isolation_forest.py
python algorithms/supervised/xgboost.py

# 2b) or run everything + build the comparison
python main.py                 # all 17 models, then the comparison table
python main.py --skip-deep     # skip the slow deep-learning models
python main.py --family supervised
python main.py --only isolation_forest,pca,random_forest

# 3) comparison table (aggregate saved metrics, or --run to retrain + measure memory)
python compare_models.py
python compare_models.py --run

# 4) per-anomaly-type coverage + which models work best together (+ graphs)
python analyze_results.py
```

Requires Python 3.10+. Deep models use PyTorch (CPU is fine for this prototype;
install a CUDA build for GPU). `xgboost` / `lightgbm` / `catboost` are needed
only for their respective supervised scripts, which degrade gracefully with an
install hint if absent.

---

## 5. Results (seeded, reproducible)

Threshold chosen per model for best raw F1; **PR-AUC** and **PA-F1** are the
most informative columns at a ~5 % base rate. These are the **honest,
alarm-free** numbers — the `alarm` bit is *not* a feature (see §1 "No label
leakage"); it is instead the baseline to beat.

| model | family | precision | recall | F1 | ROC-AUC | PR-AUC | PA-F1 |
|---|---|---|---|---|---|---|---|
| isolation_forest | unsup | 0.805 | 0.472 | 0.595 | 0.854 | 0.575 | 0.836 |
| dbscan | unsup | 0.552 | 0.402 | 0.465 | 0.712 | 0.315 | 0.860 |
| pca | unsup | 0.705 | 0.447 | 0.547 | 0.769 | 0.372 | 0.915 |
| kmeans | unsup | 0.675 | 0.460 | 0.547 | 0.806 | 0.411 | 0.901 |
| **gmm** | unsup | 0.756 | 0.657 | **0.703** | 0.839 | 0.564 | 0.867 |
| random_forest | sup | 0.566 | 0.557 | 0.561 | 0.865 | 0.530 | 0.620 |
| xgboost | sup | 0.734 | 0.399 | 0.517 | 0.665 | 0.436 | 0.717 |
| lightgbm | sup | 0.888 | 0.418 | 0.568 | 0.756 | 0.495 | 0.756 |
| catboost | sup | 0.814 | 0.323 | 0.463 | 0.693 | 0.397 | 0.503 |
| logistic_regression | sup | 0.816 | 0.444 | 0.575 | 0.723 | 0.435 | 0.737 |
| svm | sup | **0.904** | 0.359 | 0.514 | 0.672 | 0.420 | 0.767 |
| autoencoder | unsup | 0.592 | 0.347 | 0.438 | 0.707 | 0.246 | 0.750 |
| lstm_unsupervised | unsup | 0.592 | 0.403 | 0.479 | 0.787 | 0.301 | 0.769 |
| transformer | unsup | 0.545 | 0.500 | 0.522 | 0.834 | 0.370 | 0.713 |
| vae | unsup | 0.610 | 0.386 | 0.473 | 0.799 | 0.314 | 0.791 |
| usad | unsup | 0.596 | 0.349 | 0.440 | 0.712 | 0.245 | 0.750 |
| **omnianomaly** | unsup | 0.841 | 0.493 | 0.621 | 0.836 | **0.596** | **0.929** |
| gan | unsup | 0.325 | 0.190 | 0.240 | 0.523 | 0.128 | 0.573 |

Regenerate with `python main.py`; per-model figures land in `figures/`, the
table in `figures/comparison.csv` and `figures/comparison.png`. Per-anomaly-type
and best-combination analysis: `python analyze_results.py` → see `ANALYSIS.md`.

**What the numbers say** (and note how they changed once the circular `alarm`
feature was removed):
- **Removing `alarm` exposed how much the supervised models leaned on it.**
  Random Forest fell from F1 0.72 → 0.56 and XGBoost 0.70 → 0.52 — they had been
  reading the detector's own alarm bit. The unsupervised models barely moved.
  This is the whole point of the realism fix: the earlier numbers were flattered
  by a circular feature.
- **GMM is now the best single model** (F1 0.703): full-covariance density
  handles both spikes and slow drifts, and it needs no labels.
- **OmniAnomaly leads the deep models** (PR-AUC 0.60, PA-F1 0.93): its learned
  per-channel variance converts to precision on noisy SCADA-like channels.
- **Per-point methods (DBSCAN) catch short spikes** that window-aggregating deep
  models dilute; sequence models (Transformer/LSTM) catch the contextual/
  collective/cascading faults the spatial methods miss. Neither family dominates
  → the case for a **hybrid** (see ALGORITHMS.md and ANALYSIS.md).
- **`contextual` and `persistent` anomalies cap everyone at ~0.5 recall** — the
  genuine hard frontier (values stay in-band).
- **GAN is weakest** — the honest, expected result; better used for data
  augmentation than direct detection.

---

## 6. Deployment on Apache Spark (next phase)

Every detector here is, after training, a **pure function**
`score(window) → float` plus a frozen artifact (weights / reference set / mixture
parameters). Nothing but stateless artifacts and summary statistics needs to
cross between workers, so the classic streaming topology applies with no
algorithmic changes — exactly what the distributed `distributedAlgorithm14`
repo demonstrates.

**Recommended architecture (mirrors that repo + the proposal's modules):**

```
Oracle DCS (EVENTHISTORY, ojdbc8)
        │  ETL (Spark JDBC read → parquet on HDFS)      ← etl_job.py
        ▼
   HDFS parquet ──► VectorAssembler + StandardScaler
        │
        ├─ row-wise shards  → i.i.d. models (IF, KMeans, GMM, PCA, DBSCAN, AE, VAE, USAD, GAN)
        └─ sequential shards → temporal models (LSTM, OmniAnomaly)  [preserve ID order]
        │
   Distributed training/scoring:
     • MLlib native:  KMeans, GaussianMixture, PCA        (already distributed)
     • sklearn @ partitions: IsolationForest / DBSCAN via mapInPandas
     • PyTorch DDP:  AE, VAE, GAN, USAD, Transformer, LSTM via TorchDistributor
     • boosting:     LightGBM on Spark (synapseml), xgboost.spark
        │
        ▼
   Per-model anomaly flags → ensemble vote (majority) + adaptive threshold (POT/SPOT)
        ▼
   TimescaleDB/parquet → Grafana dashboard + alerting
```

**Concrete recommendations.**

1. **Partition by channel group, never round-robin.** A window must contain all
   channels of a group, and any streaming threshold state (POT/SPOT) has a
   single owner per key.
2. **Two shard strategies.** Row-wise (random) shards for i.i.d. models; strictly
   **sequential** (contiguous `element_id` + time) shards for LSTM/OmniAnomaly so
   sliding windows keep temporal order — the one non-negotiable ordering
   constraint (`Window.orderBy` once, then cache).
3. **Trees distribute for free.** Isolation Forest / Random Forest: train small
   forests per partition and ensemble-average scores at inference.
4. **Deep models via `TorchDistributor`.** Rank-specific shard reads; only
   trained weights return to the driver, then broadcast for scoring with a
   `pandas_udf`. Batch windows per partition for throughput.
5. **Adaptive thresholds, not constants.** Calibrate a POT/SPOT threshold on
   healthy data per (group, model) — a target false-alarm budget ("one per
   week") instead of a hand-tuned score cutoff. Optionally learn the FP/FN cost
   trade-off with an RL agent driven by operator acknowledgement latency
   (`ACK_TIMESTAMP` already exists in the DCS `EVENT_ALERTS` schema).
6. **Version every artifact together** (model + scaler + threshold state) — a
   scaler/model mismatch fails silently with plausible-looking scores.
7. **Deadband handling.** Real DCS values arrive on-change; forward-fill onto the
   model's time grid *before* windowing or window shapes vary.
8. **Retraining cadence** is the real ops problem (non-stationarity): schedule
   per-group retrains from recent healthy data, validate on a held-back healthy
   slice, roll out via a registry with rollback.

**Hybrid recommendation for production.** Run a cheap per-timestep screen
(Isolation Forest or DBSCAN) *and* a temporal deep branch (LSTM- or
OmniAnomaly-style), fuse them under a POT/SPOT adaptive threshold. Neither the
spatial nor the temporal family dominates alone — the hybrid covers both the
point-spike and the contextual/collective blind spots (see `ALGORITHMS.md`).

---

## 7. Notes & caveats

- The ~5 % anomaly rate and best-F1 thresholding are prototype conventions for a
  fair cross-model comparison; production uses healthy-data-calibrated
  thresholds.
- Synthetic data validates *algorithms*, not real detector physics. The loaders
  are one CSV/JDBC swap away from real DDT batches.
- All results are seeded and reproducible on CPU.

---

## 8. References (real ATLAS / CERN anomaly detection)

Which of these algorithms are actually used at ATLAS is discussed in
`ALGORITHMS.md`. Key sources:

- **DeepHYDRA** — hybrid DBSCAN + deep-learning time-series anomaly detection,
  projected for the ATLAS High-Level Trigger (TDAQ). ACM ICS 2024.
  <https://dl.acm.org/doi/10.1145/3650200.3656637> ·
  <https://arxiv.org/html/2405.07749v1>
- **Autoencoder (LSTM) anomaly detection for the ATLAS Liquid Argon calorimeter
  data quality monitoring.** <https://arxiv.org/pdf/2512.05977>
- **Synthetic data generation (Lorenzetti) for time-series anomaly detection at
  the LHC.** <https://arxiv.org/html/2509.07451>

Summary of deployment status by method: **DBSCAN, LSTM-AE, Transformer-AE,
autoencoders, Isolation Forest** appear in real ATLAS/CERN DQM/TDAQ work;
**gradient-boosted trees** are used in ATLAS physics (not DCS); **USAD,
OmniAnomaly, VAE, GAN, K-Means, GMM, PCA** are literature/benchmark candidates.
