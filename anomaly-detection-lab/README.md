# Anomaly Detection Lab — ATLAS DCS Major Project

Runnable reference implementations of every anomaly-detection family from the
literature review (DAQ@LHC workshop taxonomy), exercised on a synthetic
ATLAS-DCS-like telemetry stream. One script per technique, each producing a
figure in `figures/` and metrics in `results.json`.

```bash
pip install -r requirements.txt
python run_all.py            # everything + comparison table  (~5-10 min, CPU)
python 04_lstm_autoencoder.py  # or any single technique
```

## The synthetic data (`common/data_generator.py`)

5 correlated channels of a fictional subdetector cooling loop, 1-minute
sampling (`pixel_temp_C`, `lar_pressure_bar`, `lv_voltage_V`, `hv_current_uA`,
`cooling_flow_lpm`). **Train split (8000 steps) is healthy**; models learn
"normal" from it. **Test split (4000 steps)** contains five injected faults
mapping to the textbook anomaly types:

| id | fault | anomaly type | where |
|----|-------|--------------|-------|
| A1 | temp / HV spikes | point | 300, 1500, 3200 |
| A2 | cooling valve degrades → temp ↑ → HV ↑ → LV sags (the PrintCERN "chain failure") | persistent + collective | 800–1100 |
| A3 | temp daily cycle phase-shifted; every value in normal range, only timing wrong | contextual | 1900–2100 |
| A4 | fast HV oscillation burst, mean unchanged | collective | 2600–2750 |
| A5 | slow pressure sensor drift | trend | 3400–3900 |

## The techniques

| script | family | one-line idea | anomaly score |
|--------|--------|---------------|---------------|
| `01_dbscan.py` | clustering | healthy states are dense; noise points (-1) are outliers. T-DBSCAN = per-time-bucket DBSCAN (DeepHYDRA's spatial half) | k-NN distance to healthy train states |
| `02_isolation_forest.py` | clustering | anomalies are isolated in fewer random splits → shallow leaves | mean path length (negated) |
| `03_deep_embedding.py` | clustering + DL | autoencode whole windows, cluster in the latent space so temporal shape survives | k-NN distance between latents |
| `04_lstm_autoencoder.py` | deep learning | LSTM compresses a window into one state and must rebuild it; wrong *timing* reconstructs badly (proposal's deep half) | reconstruction MSE |
| `05_dagmm.py` | deep learning | autoencoder + GMM trained *jointly*; latent shaped so density estimation works | GMM sample energy −log p(z) |
| `06_usad.py` | deep learning | twin AEs trained adversarially; AE2 amplifies AE1's defects — cheap and sharp | α‖x−AE1(x)‖ + β‖x−AE2(AE1(x))‖ |
| `07_omnianomaly.py` | deep learning | stochastic GRU-VAE: models normality as a *distribution* per step | reconstruction NLL |
| `08_transformer.py` | deep learning | self-attention sees the whole window at once; masked-reconstruction training | reconstruction MSE |
| `09_vae.py` | generative | prior anchors "normal" in latent space | MC reconstruction probability |
| `10_gan.py` | generative | generator spans only the normal manifold; latent search fails on anomalies (AnoGAN) | residual + discriminator-feature distance |
| `11_pot_adaptive_threshold.py` | thresholding | EVT: fit a GPD to score excesses → threshold from a *target false-alarm rate*; SPOT updates it online | (turns any score into alarms) |
| `12_rl_threshold_exploration.py` | RL (toy) | Q-learning agent raises/lowers the threshold; ε-greedy = active exploration | (learned alarm policy) |

## Reading the results

- Every figure has the same top two panels: the telemetry with true anomalies
  shaded, and the score with threshold + detections — so techniques can be
  compared at a glance. The third panel is method-specific (cluster plane,
  attention map, GPD tail fit, learning curve, ...).
- `results.json` / the `run_all.py` table report point-wise P/R/F1 at the best
  raw-F1 threshold, plus the point-adjusted F1 (SMD protocol: a segment counts
  as caught if any of its points is flagged).
- Instructive comparisons: per-point spatial methods (01, 02) vs sequence
  models (04, 07, 08) on the **contextual** fault A3; everything vs the
  **drift** A5; and how the hybrid idea in the proposal (03 ≈ deep embedding
  + clustering) combines the two views.

## Relation to the proposal

The proposal's engine is DeepHYDRA-style: **T-DBSCAN (01) + LSTM/GRU
autoencoder (04) + USAD scoring (06) + adaptive thresholds (11)**. This lab
gives each ingredient in isolation so the hybrid design choices are visible.
The remaining scripts cover the alternatives reviewed in the DAQ@LHC survey.

Next step for the distributed-systems demo: wrap each detector as a worker
service consuming windows from a queue and emitting scores — the scripts are
already stateless after training, so they parallelise naturally.
