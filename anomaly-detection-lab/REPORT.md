# Technical Report — Anomaly Detection Lab for the ATLAS DCS Major Project

*Prepared 2026-07-14 · companion to the code in `anomaly-detection-lab/`*

---

## 1. Purpose and context

The major project ("Hybrid Machine Learning for Multi-Variate Time-Series
Anomaly Detection on Historical ATLAS DCS Data") proposes a DeepHYDRA-style
hybrid engine — T-DBSCAN + LSTM/GRU autoencoders + USAD-style scoring +
adaptive thresholds — served through a DDT-like API stack. Before committing
to that hybrid, the team needs working intuition for **every** technique the
literature review lists: what each one actually computes, what it catches,
what it misses, and what it costs.

This lab exists to build that intuition experimentally. Each of the twelve
techniques from the DAQ@LHC survey taxonomy is implemented as a standalone,
runnable script, exercised on a **common synthetic dataset with known ground
truth**, and reported with **identical figures and metrics** so the methods
can be compared side by side rather than argued about abstractly.

Secondary goal: the scripts are written so that each detector is a stateless
scoring function after training — deliberately, because the team's next step
is a **distributed-systems demo** (Section 8).

## 2. What was built

```
anomaly-detection-lab/
├── common/
│   ├── data_generator.py   synthetic DCS-like telemetry + fault injection
│   ├── utils.py            windowing, evaluation, standard result figures
│   └── torch_utils.py      shared training loop, batched inference
├── 01_dbscan.py … 12_rl_threshold_exploration.py   one technique each
├── run_all.py              runs everything, prints the comparison table
├── data/                   generated CSVs (train = healthy, test = faulted)
├── figures/                one figure per technique
└── results.json            metrics incl. per-fault detection coverage
```

Every technique script has the same skeleton: load data → standardize with
train statistics → (window) → train on healthy data only → score the test
stream → sweep thresholds → save metrics → render a three-panel figure
(telemetry with truth shaded / anomaly score with threshold and detections /
one method-specific diagnostic panel). The uniformity is the point: any
difference you see between two figures is a difference between the
*algorithms*, not between their harnesses.

## 3. The dataset and why it is synthetic

### 3.1 Why not real data

The 68 TB ATLAS DCS archive is **closed** (only the Good Run List is public;
ATLAS Open Data serves collision events, not detector telemetry), and access
goes through the DDT API with CERN credentials. More importantly for this
phase: real DCS data is **unlabeled** — you cannot compute precision/recall
against it without months of expert annotation. A synthetic set with exact
ground truth is the correct instrument for understanding and validating
algorithms; the proposal's own Testing & Validation Plan calls for exactly
this ("synthetic fault injections: sensor drifts, voltage spikes, connection
drops"). The loaders are one CSV swap away from real DDT batches later.

### 3.2 Design of the healthy signal

Five channels of a fictional subdetector cooling loop at 1-minute sampling,
built to reproduce the *statistical properties that make DCS data hard*:

| channel | behaviour | DCS property it models |
|---|---|---|
| `cooling_flow_lpm` | daily sine + noise | slow environmental cycles (non-stationarity) |
| `pixel_temp_C` | daily sine **minus 0.5×(cooling−12)** + noise | cross-channel physical coupling |
| `hv_current_uA` | **2×(temp+10)** + noise | second-order coupling (chain: cooling→temp→HV) |
| `lar_pressure_bar` | half-day sine + tiny noise | a second, different periodicity |
| `lv_voltage_V` | flat 3.30 V + tiny noise | a "boring" rail that only matters when it sags |

Train split: 8 000 healthy steps. Test split: 4 000 steps with five faults.

### 3.3 The injected faults — one per anomaly class

The DAQ@LHC taxonomy (point / contextual / persistent / collective / trend)
drove the fault design; each fault exists to expose a specific algorithmic
blind spot:

| id | steps | fault | class | designed to test |
|---|---|---|---|---|
| A1 | 300, 1500, 3200 | ±3 °C temp spikes, +15 µA HV spike | point | the easy case — everything should catch these |
| A2 | 800–1100 | cooling valve degrades → temp rises (30-step lag) → HV rises → LV sags | persistent + collective | the PrintCERN "chain failure"; slow ramps defeat naive residual detectors |
| A3 | 1900–2100 | temp daily cycle phase-shifted by π; **every individual value stays inside the healthy range** | contextual | separates temporal models from per-point models — a per-point detector sees nothing wrong with any single reading |
| A4 | 2600–2750 | fast HV oscillation, **mean unchanged** | collective | frequency-domain deviation with unchanged marginal statistics |
| A5 | 3400–3900 | slow +0.06 bar pressure drift | trend | sensor degradation — the "predictive maintenance" case |

One nuance discovered while building it: A3 is *partially* visible even to
per-point detectors, because shifting the temperature phase breaks the
temp↔HV correlation (HV keeps following the *healthy* temperature). That is
realistic — in a tightly coupled plant, purely contextual anomalies usually
leak into the correlation structure — and it is precisely why multivariate
detectors beat per-channel threshold rules on DCS-like data.

## 4. Evaluation methodology

- **Threshold selection.** Each detector outputs a continuous score. We sweep
  250 quantiles of the test scores and report the threshold that maximises
  **raw point-wise F1**. (Many papers tune on the point-adjusted metric,
  which inflates results; we tune on the honest one and *report* both.)
- **Point-adjusted (PA) metrics** (SMD/OmniAnomaly protocol): if any step of
  a true anomalous segment is flagged, the whole segment counts as detected.
  PA answers the operationally relevant question "did an alarm fire during
  the incident?", raw F1 answers "how precisely was the incident localised?".
- **Per-fault coverage**: fraction of each fault's steps flagged at the
  chosen threshold — this is what reveals each algorithm's characteristics.
- **Honesty caveats.** Test-set threshold tuning is an optimistic-but-uniform
  convention (identical for all twelve methods, so the *comparison* is fair);
  in production the threshold must come from POT (script 11) on healthy
  calibration data. The 29 % anomaly rate is far above reality; it is chosen
  so every fault class is represented, not to simulate base rates.

## 5. Results

### 5.1 Overall metrics (threshold at best raw F1; PA = point-adjusted)

| method | P | R | F1 | PA-F1 |
|---|---|---|---|---|
| 01_dbscan | 0.941 | 0.829 | 0.882 | 0.975 |
| 02_isolation_forest | 0.726 | 0.727 | 0.727 | 0.879 |
| 03_deep_embedding | 0.879 | 0.760 | 0.815 | 0.949 |
| 04_lstm_autoencoder | 0.845 | 0.804 | 0.824 | 0.931 |
| 05_dagmm | 0.787 | 0.779 | 0.783 | 0.904 |
| 06_usad | 0.793 | 0.600 | 0.683 | 0.926 |
| 07_omnianomaly | **0.968** | 0.824 | **0.890** | **0.986** |
| 08_transformer | 0.827 | **0.842** | 0.835 | 0.919 |
| 09_vae | 0.802 | 0.711 | 0.754 | 0.920 |
| 10_gan | 0.589 | 0.612 | 0.600 | 0.824 |
| 11_pot_adaptive_threshold | 0.997 | 0.668 | 0.800 | 0.999 |
| 12_rl_threshold_exploration | 0.720 | 0.750 | 0.735 | 0.873 |

(11 and 12 score a plain k-NN stream — their rows measure the *thresholding*
layer, not a new detector.)

### 5.2 Per-fault coverage (fraction of the fault's steps flagged)

A1a/A1b/A1c = the three point spikes (temp +3, temp −2.5, HV +15);
A2 = cascade; A3 = contextual phase shift; A4 = HV oscillation; A5 = drift.

| method | A1a | A2 | A1b | A3 | A4 | A1c | A5 |
|---|---|---|---|---|---|---|---|
| 01_dbscan | 1.00 | 0.80 | 1.00 | 0.97 | 0.75 | 1.00 | 0.82 |
| 02_isolation_forest | 1.00 | 0.75 | 1.00 | 0.85 | 0.71 | 1.00 | 0.66 |
| 03_deep_embedding | **0.00** | 0.82 | **0.00** | 0.97 | **0.01** | 0.50 | 0.87 |
| 04_lstm_autoencoder | 1.00 | 0.72 | 1.00 | 0.99 | 0.95 | 1.00 | 0.73 |
| 05_dagmm | **0.00** | 0.82 | 1.00 | 0.95 | 0.73 | 1.00 | 0.70 |
| 06_usad | **0.00** | 0.40 | **0.00** | 0.66 | **0.12** | 0.50 | 0.85 |
| 07_omnianomaly | 1.00 | 0.69 | 1.00 | 1.00 | 0.94 | 1.00 | 0.80 |
| 08_transformer | 1.00 | 0.83 | 1.00 | 1.00 | 0.99 | 1.00 | 0.74 |
| 09_vae | 1.00 | 0.70 | 1.00 | 0.99 | 0.96 | 1.00 | 0.53 |
| 10_gan | 1.00 | 0.60 | **0.00** | 0.83 | 0.90 | 1.00 | 0.44 |
| 11_pot_adaptive_threshold | 1.00 | 0.72 | 1.00 | 0.68 | 0.55 | 1.00 | 0.67 |
| 12_rl_threshold_exploration | 1.00 | 0.54 | 1.00 | 0.91 | 0.72 | 1.00 | 0.82 |

### 5.3 What the numbers say

- **Window aggregation dilutes point anomalies.** The flat-window models
  that score a whole window at once (03, 05, 06; GAN partially) post 0.00
  on 1–2-step spikes: a 2-step spike contributes 2/160 of a window's mean
  error and vanishes. Per-timestep methods (01, 02) and models that score
  the *last step* with full temporal context (04, 07, 08) catch every spike.
  This is the single strongest empirical argument for the proposal's hybrid:
  **DeepHYDRA keeps a per-timestep T-DBSCAN branch precisely because its
  windowed deep branch has this blind spot.**
- **Temporal faults separate the families the other way.** On the
  oscillation burst A4 (mean unchanged), spatial methods manage 0.71–0.75
  while sequence models post 0.94–0.99; on the contextual A3 the sequence
  models are perfect. (A3 is *partially* visible to spatial methods only
  because the phase shift breaks the temp↔HV correlation — see §3.3.)
- **Nobody fully covers a slow ramp at a point-wise threshold.** A2 and A5
  coverages of 0.7–0.85 reflect the undetectable early portion of every ramp
  (the first steps of a drift are genuinely inside normal variation). The
  PA-F1 column tells the operational story: every method fires *during*
  every long incident (PA ≈ 0.88–0.99); they differ in how early and how
  completely.
- **OmniAnomaly wins overall** (F1 0.890, P 0.968) — the learned per-channel
  noise model converts to precision. The transformer is the best balanced
  (recall 0.842). USAD's low raw recall with high PA-F1 (0.926) means it
  fires reliably but briefly — acceptable for alerting, poor for incident
  extent estimation, and tunable via its α/β knob.
- **SPOT delivers 0.997 precision with no test-set tuning** — calibrated
  purely on healthy data, which none of rows 01–10 can claim (their
  thresholds are tuned on the test scores; §4). This is the fairest preview
  of production behaviour in the table.

## 6. The algorithms in depth

Throughout this section, "DCS fit" judges each method against the properties
that define the ATLAS DCS problem: **(a)** millions of correlated channels,
**(b)** non-stationary baselines (LHC run cycles, seasons, configuration
changes), **(c)** cascading multi-channel faults, **(d)** SCADA deadband
logging (values recorded on change or on a slow heartbeat → irregular
sampling), **(e)** no labels, **(f)** historical batch access through an API
rather than a live stream, **(g)** operators who must be able to interpret an
alarm.

### 6.1 DBSCAN — density-based clustering (`01_dbscan.py`)

**Mechanics.** Each timestep is a point in R⁵ (the standardized sensor
vector). DBSCAN groups points that have at least `min_samples` neighbours
within radius `eps` into clusters; points reachable from no dense core are
labelled noise (−1). There is no "number of clusters" parameter and no
assumption of cluster shape — density is the only criterion. Healthy
operation traces out a dense manifold (the daily cycle sweeps a curved,
correlated path through sensor space); faults leave that manifold.

**Two refinements used here.** (1) *T-DBSCAN*: clustering is run per
500-step time bucket rather than globally, as in DeepHYDRA — a state is
judged dense *relative to its local time context*, which tolerates slow
drift of the operating point. (2) Because DBSCAN's −1/cluster output is
binary and threshold sweeps need a continuous score, the reported score is
the distance to the k-th nearest healthy *training* state — the same
quantity DBSCAN internally compares against `eps`.

**Behaviour observed.** Excellent on point spikes and the cascade (states
far off the manifold), good on the drift once it exceeds the healthy
envelope, weakest on the parts of A3/A4 where the multivariate value stays
inside dense regions. No training beyond storing the reference set.

**DCS fit.** Cheap, explainable ("this state is 4.2σ from any state seen in
the last 3 days"), trivially parallel per bucket. But per-*timestep*: it
cannot see temporal shape at all, and k-NN queries against a large reference
set need an index (KD-tree/FAISS) to scale. In DeepHYDRA it is the fast
first-pass filter, not the whole answer — the lab reproduces that logic.

### 6.2 Isolation Forest (`02_isolation_forest.py`)

**Mechanics.** Build many random binary trees: at each node pick a random
feature and a random split value. A point's *path length* — how many splits
isolate it from the rest — is short for outliers (they sit alone in sparse
regions, so random splits cut them off quickly) and long for inliers. Score
= expected path length over the forest, normalised. Trained on healthy data;
`max_samples=1024` subsampling per tree is not a shortcut but part of the
algorithm (it sharpens isolation contrast).

**Behaviour observed.** Solid all-rounder on out-of-range faults; like
DBSCAN it is blind to purely temporal structure. Slightly worse than the
k-NN score here because axis-parallel splits approximate the tilted,
correlated healthy manifold coarsely — a characteristic weakness on strongly
correlated channels.

**DCS fit.** O(n log n) training, microsecond scoring, no distance metric to
tune, naturally embarrassingly parallel (trees are independent — they can be
trained and evaluated on different workers). Used at CERN (ALICE/IT cloud
monitoring). Good as a cheap always-on screen across thousands of channel
groups; not the tool for cascades' temporal signature.

### 6.3 Deep Embedding + Clustering (`03_deep_embedding.py`)

**Mechanics.** The bridge between the clustering family and deep learning.
A dense autoencoder compresses each 32-step × 5-channel *window* (160
values) to an 8-d latent vector; k-NN distance (and DBSCAN, for the picture)
is then computed *between latents*. Because each latent summarises a whole
window, temporal shape survives into the density reasoning — raw per-point
clustering throws it away.

**Why it matters conceptually.** This is DeepHYDRA's core design idea in
miniature: *let a neural network define the space, let a density method make
the decision*. The clustering stage stays explainable and thresholdable; the
network absorbs the nonlinearity.

**Behaviour observed — a caution.** The per-fault table (§5.2) shows the
limits of a *dense* encoder: it caught the cascade, contextual, and drift
faults well but scored 0.00 on point spikes (window dilution, §5.3) and 0.01
on the oscillation burst — a flat MLP over a flattened window has no
inductive bias for frequency content, and the AE's compression discarded it
as noise. "Deep embedding" is a family, not a guarantee: swap the dense
encoder for the LSTM (04) or transformer (08) encoder and cluster *those*
latents to keep the explainable-decision layer without these blind spots.

**DCS fit.** Latents are 20× smaller than windows — for DCS scale this is
also a *bandwidth* statement: ship 8 floats per window between services, not
160. Training is unsupervised (property e). Weakness: the latent geometry is
only as good as the AE's training distribution; a configuration change
(property b) moves the manifold and requires re-training or adaptive
recalibration.

### 6.4 LSTM Autoencoder (`04_lstm_autoencoder.py`)

**Mechanics.** An LSTM (Long Short-Term Memory) cell maintains a memory
vector regulated by three learned gates (input/forget/output), which lets
gradients survive across long sequences — the failure mode of plain RNNs.
The encoder LSTM reads the window step by step and ends with a single hidden
state; the decoder LSTM must reproduce the entire window from that state
alone (zero-input decoding — no teacher forcing, so the bottleneck is real).
Score = mean squared reconstruction error. Trained on healthy windows only,
the model internalises the *temporal grammar*: what follows what, at which
phase, with which cross-channel lag.

**Behaviour observed.** The figure's third panel is the key exhibit: during
A3 the reconstruction follows the *healthy* phase while the actual signal
follows the shifted one — a visible gap despite every value being in range.
This is exactly the fault class per-point methods cannot represent.

**DCS fit.** Directly matches property (c): cascades are temporal sequences
(cooling→temp lag of ~30 steps is learnable). Handles (b) if the training
window covers the cycles. Costs: sequential computation (no
parallelism across time steps → slowest family per window), needs GPU at
scale, reconstruction error is less interpretable than a distance. This is
the proposal's chosen deep half, and the lab confirms it earns its place on
contextual/cascade faults specifically.

### 6.5 DAGMM — Deep Autoencoding Gaussian Mixture Model (`05_dagmm.py`)

**Mechanics.** Two networks trained **jointly**, which is the entire point.
The *compression net* (an AE) produces a low-d latent z_c plus two
reconstruction-quality features (relative Euclidean error, cosine
similarity) — so "how badly did compression fail" is itself part of the
representation z. The *estimation net* outputs soft memberships γ of z in a
K-component Gaussian mixture; mixture parameters (φ, μ, Σ) are computed from
batch statistics weighted by γ. Score = sample **energy** E(z) = −log Σₖ φₖ
N(z; μₖ, Σₖ). The loss is reconstruction + λ₁·energy + λ₂·covariance
regularisation, so the AE is explicitly shaped to make density estimation
easy — unlike an AE + separately fitted GMM, where the latent may be
arbitrarily unsuited to Gaussians. After training, (φ, μ, Σ) are frozen from
the full healthy set.

**Behaviour observed.** Competitive; catches faults through *either* door —
poor compressibility (spikes) or low latent density (drift, cascade). The
energy histogram panel shows clean tail separation. Implementation note: the
mixture energy needs Cholesky-based solves and a small diagonal jitter for
numerical stability; a broadcasting bug in exactly this code path was the
one failure in the first full run (fixed).

**DCS fit.** One elegant score with a probabilistic reading ("this state has
likelihood 10⁻⁷ under 3 months of healthy operation"). Two doors = robust to
fault diversity (property c). Cons: the most delicate training in the lab
(mixture collapse if λ's are off), and the energy is only as calibrated as
the healthy sample is representative (property b again).

### 6.6 USAD — UnSupervised Anomaly Detection (`06_usad.py`)

**Mechanics.** One shared encoder E, two decoders D1, D2, trained in a
two-phase adversarial game with an epoch-dependent weight (n = epoch):

- AE1 trains to reconstruct x **and** to make its output AE2-proof:
  loss₁ = (1/n)‖x−D1(E(x))‖² + (1−1/n)‖x−D2(E(D1(E(x))))‖²
- AE2 trains to reconstruct real x but to *amplify* the error of AE1's
  output: loss₂ = (1/n)‖x−D2(E(x))‖² − (1−1/n)‖x−D2(E(D1(E(x))))‖²

Early epochs are plain autoencoding (1/n ≈ 1); as n grows, the adversarial
terms dominate and AE2 becomes a magnifying glass for reconstruction
defects. Score = α‖x−AE1(x)‖² + β‖x−AE2(AE1(x))‖², with α/β trading false
positives against sensitivity *at inference time without retraining*. It
gets GAN-like sensitivity with autoencoder-like training stability, from
dense layers only.

**Behaviour observed.** Sharp on abrupt faults; the training-curve panel
shows loss₂ turning negative exactly as the adversarial phase engages.
Chosen conservative α=β=0.5 gives high precision and mid recall — the α/β
knob is a genuinely useful operational dial.

**DCS fit.** The DAQ@LHC survey's efficiency finding is the headline: USAD
reaches near-transformer detection at a fraction of the compute, which is
why the proposal borrows its scoring. Dense-only → fastest deep training in
the lab; window-flattened input ignores step order beyond what the window
captures (it relies on the window, not recurrence, for temporal context).

### 6.7 OmniAnomaly (`07_omnianomaly.py`)

**Mechanics.** The stochastic-recurrent idea: model normality as a
**distribution over trajectories**, not a single reconstruction. A GRU
carries temporal state; at each step the model infers a Gaussian posterior
q(z_t|x_{≤t}) (reparameterised sampling), a decoder GRU maps the sampled z
back to a Gaussian over x_t (mean *and* variance). Training maximises the
ELBO = reconstruction log-likelihood − KL(posterior ‖ prior). Score =
negative reconstruction log-likelihood of the last step, averaged over
posterior samples. (Simplified from the paper: standard-normal prior instead
of a linear Gaussian state-space model, no planar normalizing flows.)

**Why the variance matters.** A plain AE must pay the same penalty for noise
it cannot predict as for anomalies; a probabilistic decoder learns "this
channel is noisy, ±0.2 is normal" and prices deviations in *units of learned
uncertainty*. The figure's belief-band panel shows the actual cascade signal
walking out of the ±2σ band — that is the score.

**Behaviour observed.** Best raw F1 in the lab, and the highest precision
among the deep models — the learned noise model suppresses false alarms on
the noisy LV rail that trip fixed-penalty reconstruction methods.

**DCS fit.** SCADA telemetry is exactly "heterogeneous noise levels per
channel" (property d), which is this model's home turf. NLL is a calibrated,
comparable score across channel groups. Cost: the heaviest training in the
lab (~80 s here, hours at scale), sequential like all RNNs, and stochastic
scoring needs several forward passes (amortisable by batching).

### 6.8 Transformer autoencoder (`08_transformer.py`)

**Mechanics.** Self-attention computes, for every step, a weighted mixture
of *all* steps (softmax(QKᵀ/√d)·V per head) — every pairwise temporal
interaction is one matrix multiply away, versus 32 sequential updates for an
RNN. The lab uses 2 pre-norm encoder blocks (4 heads, d=32) with a learned
positional embedding (attention is permutation-invariant without it),
trained with a **denoising objective**: randomly mask 20 % of steps, require
reconstruction of the full window. Masking prevents the identity shortcut
that plain transformer autoencoders learn (they can trivially copy input to
output), forcing the model to encode real temporal structure — TranAD
achieves the same end with an adversarial second decoder.

**Behaviour observed.** Best balanced P/R among reconstruction methods; the
attention-map panel over the A4 window visibly locks onto the period-8
oscillation — attention maps are a genuine interpretability channel
(property g), not just decoration.

**DCS fit.** Parallel across the window → GPU-efficient at scale, the reason
transformers dominate long-horizon workloads. For 32-step windows, however,
the O(T²) advantage is idle; the DAQ@LHC finding that USAD-class models are
more *efficient* for hybrid pipelines matches what this lab observes
(transformer ≈ LSTM quality here at higher parameter budget). Worth it when
windows grow to hours/days of context — e.g., week-scale degradation
patterns.

### 6.9 VAE — Variational Autoencoder (`09_vae.py`)

**Mechanics.** Like the AE, but the encoder outputs a *distribution*
q(z|x) = N(μ(x), σ(x)) and the loss adds KL(q ‖ N(0,I)). The KL term is the
structural difference: it forbids the encoder from scattering codes
arbitrarily, pulling all healthy windows into one smooth, contiguous region
around the prior. The latent space has no "holes" of unassigned meaning —
which is what makes low reconstruction probability a *trustworthy* anomaly
signal rather than an artifact of a ragged code layout. Score = Monte-Carlo
reconstruction error over several posterior samples (the reconstruction-
probability estimator of An & Cho).

**Relationship to 6.7.** OmniAnomaly *is* a VAE made recurrent; script 09
isolates the generative-prior idea from the recurrence so the two
contributions can be understood separately. Comparing 09's and 07's results
directly measures what recurrence adds.

**DCS fit.** The prior gives a natural "distance from normality" geometry
and cheap sampling of synthetic healthy windows (useful for testing). As a
window model without recurrence it inherits the same contextual-fault
ceiling as USAD; per-channel decoder variance (as in 07) would be the first
upgrade for SCADA noise heterogeneity.

### 6.10 GAN — AnoGAN-style (`10_gan.py`)

**Mechanics.** A generator G: z→window and discriminator D train in the
usual minimax game until G's range approximates the manifold of *healthy*
windows (it has never seen anything else). Detection inverts the generator:
for each test window, optimise z (200 Adam steps, all windows batched) to
minimise ‖G(z)−x‖²; score = (1−λ)·residual + λ·‖f(x)−f(G(z*))‖² where f is
the discriminator's feature layer. The logic: if x is healthy, some z
reproduces it (residual ≈ 0); if x is off-manifold, *no* z can — the best
imitation still misses, and the miss is the score.

**Behaviour observed.** Weakest quantitative result in the lab, and that is
the honest, instructive outcome: GAN training on this small data is
unstable (mode coverage), and latent-search inversion is expensive and
noisy. The panel shows it directly — G imitates a healthy cooling-flow
window closely but produces a visibly wrong match for a cascade window.

**DCS fit.** Inference-time optimisation per window is a poor match for
high-throughput scoring (properties a, f); successor designs (BiGAN/f-AnoGAN,
TAnoGAN) train an encoder to replace the search. In this project's context
GANs are better used for *augmentation* — synthesising rare-fault
neighbourhoods — than as the primary detector. Included because the survey
includes it; the lab's contribution is showing *why* it isn't the pick.

### 6.11 POT / SPOT — adaptive thresholds from Extreme Value Theory (`11_pot_adaptive_threshold.py`)

**Mechanics.** Not a detector — the layer every detector needs. The
Pickands–Balkema–de Haan theorem says the distribution of *excesses over a
high threshold u* converges to a Generalized Pareto Distribution regardless
of the (unknown) parent distribution. POT: set u at e.g. the 98th percentile
of a healthy calibration stream, fit GPD(ξ, σ) to the excesses, then invert
the tail to the level z_q crossed with target probability q:
z_q = u + (σ/ξ)·((qn/N_t)^(−ξ) − 1). The operator chooses **q — a false-alarm
budget** ("one false alarm per week") — instead of eyeballing a score
threshold. SPOT is the streaming variant: each new sub-threshold excess
updates the fit, so the threshold *tracks* slow distribution drift; scores
that cross z_q raise alarms and are deliberately *not* absorbed into the fit
(anomalies must not teach the threshold).

**Behaviour observed.** On a k-NN score stream, SPOT calibrated only on
healthy data gives ~0.997 precision with zero test-set tuning — the
practical answer to "where does the threshold come from in production?",
which the F1-sweep convention of Section 4 deliberately dodges. The GPD
tail-fit panel shows measured excesses lying on the fitted curve.

**DCS fit.** Directly addresses properties (b) and (e): thresholds derived
from data, per channel group, adapting online, with an interpretable knob.
This is the proposal's Module 4 and it composes with *any* upstream score
(01–10). Near-zero compute.

### 6.12 Reinforcement learning — learned thresholds + active exploration (`12_rl_threshold_exploration.py`)

**Mechanics (toy, and labelled as such).** Frame thresholding as a
sequential decision problem. State = (current threshold multiplier bucket,
recent alarm-rate bucket); actions = lower/keep/raise; reward encodes the
operational trade-off (+1 true alarm, −0.3 false alarm, −1 miss, +0.01 quiet
true-negative). Tabular Q-learning, ε-greedy with decaying ε — the
**active exploration** bullet of the survey: with probability ε the agent
deliberately tries a non-greedy threshold to sample how other regimes pay
off, and the lab keeps the best-episode Q snapshot because tabular learning
on one trace is noisy by design.

**Where the reward would come from in reality.** The lab uses ground-truth
labels; production would use operator responses — the ATLAS `EVENT_ALERTS`
schema already stores `ACK_TIMESTAMP` per alarm, i.e., the feedback signal
(acknowledged fast = true alarm; silenced/ignored = false alarm) **already
exists in the database**. That mapping — RL reward from operator
acknowledgement latency — is the concrete, publishable idea hiding in this
toy, and the survey notes the CERN accelerator-controls team is working in
exactly this direction.

**DCS fit.** POT (6.11) is principled but assumes the *cost* of FP/FN is
encoded in q; RL learns the cost trade-off from interaction instead.
Realistically a later-phase upgrade: run POT first, add RL when alarm-
feedback data has accumulated.

## 7. Cross-cutting findings

1. **The per-point / temporal split is the fundamental divide.** Spatial
   methods (01, 02) are cheap and strong on out-of-range faults but
   structurally cannot represent "right value, wrong time" (A3) or "right
   mean, wrong frequency" (A4); sequence models (04, 07, 08) can. This is
   the empirical justification for the proposal's *hybrid*: neither family
   dominates the other.
2. **Probabilistic scoring pays.** OmniAnomaly's learned per-channel
   variance bought the best precision — on noisy SCADA-like channels, "error
   in units of expected noise" beats raw error.
3. **Efficiency findings from the literature replicate.** USAD trains in a
   fraction of the RNN/transformer budget; the transformer's extra capacity
   does not pay at 32-step windows. Matches the DAQ@LHC review's guidance.
4. **Thresholding is a first-class problem.** The same k-NN score went from
   F1 ≈ 0.01 (naive q, wrong score scale, first attempt) to F1 = 0.80 with a
   properly calibrated SPOT — more variance than most *model* choices. Any
   production pipeline needs Module-4-style adaptive thresholds, not tuned
   constants.
5. **Slow ramps defeat one-step residuals.** The first POT attempt scored
   EWMA one-step residuals: the A2 cascade ramps slowly enough that each
   step is unsurprising given the last — near-zero residual throughout. The
   fix (distance-to-healthy-manifold score) is a general lesson: *score
   against normality, not against yesterday*.
6. **GANs underperform as direct detectors** at this scale and are the only
   method whose inference requires optimisation; use them for augmentation.

## 8. Distributed-system implementation

### 8.1 Why these algorithms distribute naturally

After training, every detector in the lab is a **pure function**
`score(window) → float` (plus a frozen artifact: weights, reference set, or
GMM parameters). No inference-time state is shared between windows except in
SPOT (a small, single-owner threshold state per channel group). That means
the classic streaming topology applies with no algorithmic changes:

```
                        ┌────────────────────────────────────────────┐
                        │                control plane               │
                        │  model registry · training jobs · config  │
                        └──────┬─────────────────────────┬───────────┘
                               │ models                  │ models
┌──────────┐   windows  ┌──────▼──────┐  scores  ┌───────▼───────┐  alarms  ┌─────────┐
│ ingester │ ─────────▶ │  detector   │ ───────▶ │  aggregator + │ ───────▶ │ Grafana │
│ (DDT API │  (queue,   │  workers    │  (queue) │  SPOT thresh. │  (+ DB)  │  + API  │
│  replay) │  by group) │  ×N, ×algo  │          │  per group    │          └─────────┘
└──────────┘            └─────────────┘          └───────────────┘
```

- **Ingester** replays historical batches from the DDT-style API (the
  project explicitly targets *historical* data, so "streaming" is a replay
  at chosen speed — ideal for a demo), assembles sliding windows, publishes
  to a queue partitioned by **subdetector channel group**.
- **Detector workers** subscribe to partitions; each worker loads one model
  artifact and maps windows→scores. Horizontal scaling = add workers per
  partition. Different *algorithms* can run side-by-side on the same
  windows (one consumer group each) — which turns the lab's comparison
  table into a *live* dashboard race.
- **Aggregator** owns SPOT state per (group, algorithm), converts scores to
  alarms, writes both to TimescaleDB; Grafana reads it.
- **Control plane** retrains on schedule (new healthy baselines → property
  b) and rolls out versioned artifacts.

### 8.2 How each algorithm maps onto the workers

| algorithm | worker artifact | inference cost/window | distribution notes |
|---|---|---|---|
| DBSCAN/k-NN (01) | reference matrix + KD/FAISS index | one k-NN query | shard the index by time bucket or group; queries are read-only → replicate freely |
| Isolation Forest (02) | serialized forest | ~µs tree walks | trees are independent → even a single model can be split across workers, though replication is simpler |
| Deep embedding (03) | encoder weights + latent index | 1 tiny MLP + k-NN | encoder on the worker, latent index shardable; latents (8 floats) are what crosses the network — 20× bandwidth saving |
| LSTM-AE (04) | weights | 1 recurrent pass | batch windows per partition for GPU efficiency; stateless between windows |
| DAGMM (05) | weights + (φ, μ, Σ) | 1 MLP + energy eval | energy is closed-form given frozen mixture — pure function |
| USAD (06) | E, D1, D2 weights | 2 dense passes | cheapest deep scorer → the right default for wide fan-out across many groups |
| OmniAnomaly (07) | weights | k sampled passes | the k samples of one window are independent → parallel inside the worker |
| Transformer (08) | weights | 1 parallel pass | best GPU batch efficiency; give it its own batched-GPU worker pool |
| VAE (09) | weights | k dense passes | same as 07 without recurrence |
| GAN (10) | G, D weights | 200-step optimisation | the outlier: keep off the hot path; run as an async "second-opinion" service on windows already flagged by cheap detectors |
| POT/SPOT (11) | GPD params + tail buffer | trivial | *stateful but tiny*: one owner per (group, algo) key — natural fit for a keyed-state stream processor or a single aggregator consumer per partition |
| RL threshold (12) | Q table / policy | trivial | learns *in* the aggregator from operator ACK feedback; policy updates are per-key like SPOT |

Training distributes on a different axis: **by channel group** (each group's
model is independent — embarrassingly parallel across a cluster, matching
the proposal's "Silicon Pixel / LAr Cryogenics" grouping), and only within
one group's deep model would data-parallel GPU training (DDP) ever be
needed.

### 8.3 Concrete demo plan (maps to the proposal's modules)

Docker Compose, ~6 services, all components already exist in the lab:

1. **`redis`** (or Kafka if you want the résumé line; Redis Streams with
   consumer groups is lighter and sufficient) — the queue.
2. **`ingester`** — Python: replays `data/dcs_test.csv` at ×60 speed,
   publishes windows to `windows:<group>`. (Later: swap the CSV read for a
   DDT API `GET`.) — *Module 1.*
3. **`worker-usad`, `worker-lstm`, `worker-knn`** — each wraps the
   corresponding lab script's model in a ~50-line consumer loop; each is a
   separate consumer group so all three score every window. Scale test:
   `docker compose up --scale worker-usad=4`. — *Modules 2+3.*
4. **`aggregator`** — consumes scores, runs SPOT per (group, algo), writes
   scores+alarms to TimescaleDB hypertables. — *Module 4.*
5. **`timescaledb` + `grafana`** — provisioned dashboard: telemetry overlay,
   three score lanes, adaptive threshold line, alarm annotations; a second
   panel shows per-algorithm alarm latency on the same faults — the live
   version of this report's comparison table.

Demo script for the evaluation panel: start the stack, watch the cascade
(A2) arrive; k-NN fires first (per-point, no window lag), LSTM confirms with
the temporal signature, SPOT threshold visibly rides the noise floor; kill
one USAD worker mid-run to show consumer-group rebalancing (fault tolerance
in one command). Measure and report the proposal's own metrics: end-to-end
alarm latency, windows/second per worker, scaling curve vs worker count.

### 8.4 Design cautions for the real system

- **Partition by channel group, never round-robin**: windows must contain
  *all* channels of a group, and SPOT state has a single owner per key.
- **Version every artifact** (model + scaler + threshold state trained
  together); a scaler/model mismatch fails silently with plausible scores.
- **Backpressure**: historical replay can outrun deep workers; bounded
  streams + lag monitoring, and let the cheap detectors (01/02/06) stay
  real-time while heavier models trail.
- **Deadband data**: real DCS values arrive on-change; the ingester must
  forward-fill onto the model grid *before* windowing, or window shapes vary.
- **Retraining cadence** is the real ops problem (non-stationarity):
  schedule per-group retrains from recent healthy data, validate on a
  held-back healthy slice, roll out via the registry with rollback.

## 9. Recommended next steps

1. Build the Section 8.3 compose demo (USAD + LSTM-AE + k-NN + SPOT — i.e.,
   exactly the proposal's hybrid, distributed).
2. Validate the same scripts on a public labelled benchmark (SMD or
   SMAP/MSL) to anchor against published USAD/OmniAnomaly numbers.
3. Prototype the DDT-shaped FastAPI facade over TimescaleDB now (Module 1)
   so the ingester's only future change is a base URL.
4. Pursue the RL-from-`ACK_TIMESTAMP` idea (6.12) as the project's novel
   contribution beyond reproduction.

---

*All numbers in Section 5 regenerate with `python run_all.py` (seeded, CPU,
~6 min). Per-script documentation lives in each file's docstring; figures in
`figures/`.*
