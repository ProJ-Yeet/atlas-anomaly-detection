# Algorithm Guide — ATLAS DCS Anomaly Detection

This document explains **what each algorithm does**, **what it is good at**, and
**how it fits into an ATLAS Detector Control System (DCS) anomaly-detection
system**. Algorithms are ordered by implementation difficulty — the same order
in which they were built and the recommended order in which to study them.

The "DCS fit" of every method is judged against the properties that define the
real problem:

- **(a)** millions of correlated channels;
- **(b)** non-stationary baselines (LHC run cycles, seasons, reconfigurations);
- **(c)** cascading multi-channel faults;
- **(d)** SCADA deadband logging (irregular sampling, per-channel noise levels);
- **(e)** no labels on real data;
- **(f)** historical batch access through an API, not a live stream;
- **(g)** operators who must be able to interpret an alarm.

A recurring theme runs through the whole guide — the **per-point vs temporal
divide**: spatial methods (Isolation Forest, DBSCAN, PCA, K-Means, GMM) are
cheap and strong on out-of-range faults but *structurally cannot* represent
"right value, wrong time" (contextual) or "right mean, wrong frequency"
(collective) anomalies; sequence models (LSTM, Transformer, OmniAnomaly) can.
Neither family dominates — which is the empirical argument for a **hybrid**.

---

## Unsupervised detectors

These train **without labels** — the honest setting for real DCS data. In this
prototype they are scored against injected ground truth for evaluation only.

### 1. Isolation Forest — `algorithms/unsupervised/isolation_forest.py`
**What it does.** Builds many random binary trees; a point's *path length* to
isolation is short for outliers (sparse regions cut off quickly) and long for
inliers. Score = normalised mean path length.
**Good at.** A cheap, robust, all-round baseline; no distance metric to tune;
`O(n log n)` training, microsecond scoring.
**DCS fit.** Trees are independent → embarrassingly parallel on Spark (the
distributed reference implementation trains one small forest per partition and
ensemble-averages). Blind to temporal shape, so use it as an always-on *screen*
across thousands of channel groups, not the whole answer. **Start here.**

### 2. DBSCAN / density — `algorithms/unsupervised/dbscan.py`
**What it does.** Density clustering: dense regions are "normal", points
reachable from no dense core are anomalies. For a continuous score we use the
distance to the *k-th nearest healthy neighbour* (the quantity DBSCAN compares
against `eps`), with `eps` set to a high percentile of reference k-NN distances.
**Good at.** No assumed cluster shape, no "number of clusters"; interpretable
("this state is far from anything seen recently").
**DCS fit.** Shardable per time bucket / channel group; k-NN queries need an
index (KD-tree/FAISS) to scale. Per-timestep → temporally blind. The fast
first-pass filter in a DeepHYDRA-style hybrid.

### 3. PCA reconstruction — `algorithms/unsupervised/pca.py`
**What it does.** Projects onto the top principal components and reconstructs;
healthy, tightly-correlated telemetry reconstructs almost perfectly, anomalies
land off the subspace → large **reconstruction error** = score.
**Good at.** Near-free to train, closed-form, and you can read off *which*
channels drove the residual. The linear ancestor of every autoencoder here.
**DCS fit.** Native to Spark MLlib; an excellent sanity-check baseline. Linear,
so it cannot model nonlinear manifolds or temporal shape.

### 4. K-Means — `algorithms/unsupervised/kmeans.py`
**What it does.** Partitions into `k` clusters (healthy operating regimes) and
scores each row by distance to its nearest centroid.
**Good at.** Fast, native to Spark MLlib, interpretable ("nearest normal regime
is 5.3 units away").
**DCS fit.** Trivially distributed. Assumes convex, roughly spherical clusters →
weaker than DBSCAN on curved manifolds; temporally blind.

### 5. Gaussian Mixture Model — `algorithms/unsupervised/gmm.py`
**What it does.** Models healthy data as a mixture of Gaussians; score =
negative log-likelihood under the mixture.
**Good at.** Elliptical, correlated clusters (full covariance) and a *calibrated
probability* rather than a raw distance ("this state has probability 1e-7 under
normal operation").
**DCS fit.** Native to Spark MLlib; interpretable likelihood. Assumes Gaussian
components; still per-timestep.

### 8. Dense Autoencoder — `algorithms/unsupervised/autoencoder.py`
**What it does.** A neural net reconstructs each window through a narrow
bottleneck; trained on (mostly-healthy) windows, it rebuilds normal telemetry
well and anomalies poorly → reconstruction error = score. The nonlinear
generalisation of PCA.
**Good at.** Capturing nonlinear channel couplings PCA misses; fully
unsupervised.
**DCS fit.** After training it is a pure `score(window) → float` that shards via
`TorchDistributor`. A *flattened* dense AE has no temporal inductive bias, so it
dilutes short spikes across the window — the recurrent/attention variants fix
this.

### 9. LSTM Autoencoder — `algorithms/unsupervised/lstm_unsupervised.py`
**What it does.** Encoder/decoder LSTMs reconstruct a window while modelling
**temporal order** (what follows what, at which phase, with which cross-channel
lag). Reconstruction error = score.
**Good at.** *Contextual* faults (right value, wrong time) and lagged cascades
that per-point methods cannot see.
**DCS fit.** Directly matches property (c). This is the deep half of the
proposal's hybrid. Cost: sequential computation → slowest family per window,
wants a GPU at full scale.

### 10. Transformer Autoencoder — `algorithms/unsupervised/transformer.py`
**What it does.** Self-attention relates every pair of timesteps in one parallel
operation; a denoising objective (input dropout) forces it to encode real
temporal structure rather than copy the input. Reconstruction error = score.
**Good at.** Long-range dependencies, collective/frequency faults (attention
locks onto periodicity), and interpretability via attention maps (property g).
**DCS fit.** Parallel across the window → GPU-efficient. For short windows its
`O(T²)` advantage is idle; it earns its keep at hour/day-scale context.

### 11. Variational Autoencoder — `algorithms/unsupervised/vae.py`
**What it does.** Like a dense AE but the encoder outputs a *distribution*
`q(z|x)` and the loss adds a KL term pulling it toward a standard-normal prior.
That regularisation removes latent "holes", so low reconstruction probability
becomes a *trustworthy* anomaly signal. Score = Monte-Carlo reconstruction
error.
**Good at.** A principled "distance from normality" geometry; can sample
synthetic healthy windows for testing.
**DCS fit.** The static ancestor of OmniAnomaly — comparing the two isolates
what recurrence adds. As a window model it shares the contextual-fault ceiling
of non-recurrent methods.

### 12. USAD — `algorithms/unsupervised/usad.py`
**What it does.** One shared encoder feeds two decoders in a two-phase
adversarial game: AE2 learns to *amplify* AE1's reconstruction error, giving
GAN-like sensitivity with autoencoder-like stability, from dense layers only.
Score = `α·‖x−AE1(x)‖² + β·‖x−AE2(AE1(x))‖²` — the α/β knob trades false
positives against sensitivity **at inference time without retraining**.
**Good at.** Sharp on abrupt faults at a fraction of the compute of RNNs/
transformers — the survey's efficiency headline, which is why the proposal
borrows its scoring.
**DCS fit.** Cheapest deep scorer → the right default for wide fan-out across
many channel groups. *Implementation note:* the adversarial `−mse` term is
unbounded; training must use one shared forward pass, back-prop both losses,
then step both optimizers together (plus gradient clipping) — recomputing the
forward on half-updated weights diverges and inverts the score.

### 13. OmniAnomaly (simplified) — `algorithms/unsupervised/omnianomaly.py`
**What it does.** Stochastic-recurrent model: a GRU carries temporal state, a
Gaussian latent is inferred per step, and a decoder emits a Gaussian over each
observation (mean **and** variance). Trained by maximising the ELBO; score =
negative reconstruction log-likelihood of the last step.
**Good at.** The learned per-channel variance prices deviations in *units of
expected noise*, which typically wins **precision** on heterogeneous SCADA
channels — confirmed here (best deep-model precision/PR-AUC).
**DCS fit.** SCADA telemetry *is* "heterogeneous noise per channel" (property
d) — this model's home turf. NLL is a calibrated, comparable score across
channel groups. Cost: the heaviest training here, sequential like all RNNs.

### 14. GAN-based (simplified AnoGAN) — `algorithms/unsupervised/gan.py`
**What it does.** A generator learns the manifold of *healthy* windows; an
anomaly is off-manifold, so the discriminator's realness estimate is low. Score
= blend of the discriminator's negative logit and a feature-matching residual.
**Good at.** Little, as a *direct* detector on small data (unstable training,
incomplete mode coverage) — the honest, instructive outcome and the weakest
method here.
**DCS fit.** The canonical AnoGAN inverts the generator per window
(optimisation loop per sample) — a poor match for high-throughput scoring.
GANs are better used to **augment** rare-fault data than to detect. Included for
completeness because the survey lists it.

---

## Supervised classifiers

These use the **labeled** dataset. Evaluation uses a *temporal* train/test split
(first 60 % of each element's timeline trains, last 40 % tests) — never a random
shuffle, which would leak the future into the past. All supervised models share
one caveat: they recognise the fault *types seen in training*, not novel ones —
so pair them with an unsupervised screen.

### 6. Random Forest — `algorithms/supervised/random_forest.py`
**What it does.** An ensemble of decision trees on bootstrap samples; class
probability = fraction of trees voting "anomaly". `class_weight="balanced"`
handles the ~5 % positive rate.
**Good at.** A strong, robust supervised baseline; exposes feature importances
(property g); parallel across trees.
**DCS fit.** Far more precise than unsupervised methods when labels exist (here
P≈0.88). The natural first supervised model.

### 7. XGBoost — `algorithms/supervised/xgboost.py`
**What it does.** Gradient-boosted trees built sequentially, each correcting the
ensemble's residual errors; `scale_pos_weight` rebalances the classes.
**Good at.** Top-tier accuracy on engineered tabular features (highest precision
here, P≈0.92); handles missing values natively.
**DCS fit.** Distributes on Spark via `xgboost.spark`. The project's own ranking
puts XGBoost with engineered features at #1.

### 7. LightGBM — `algorithms/supervised/lightgbm.py`
**What it does.** Gradient boosting with histogram binning and leaf-wise tree
growth → markedly faster/lighter than XGBoost at similar accuracy.
**Good at.** Speed on large tabular data (matters at 68 TB), native categorical
handling.
**DCS fit.** Spark connector via `synapseml` / `lightgbm.dask`. Best
speed/accuracy trade-off among the boosters here.

### 7. CatBoost — `algorithms/supervised/catboost.py`
**What it does.** Gradient boosting with *ordered boosting* (reduces
prediction-shift bias) and first-class categorical support; symmetric trees.
**Good at.** Excellent defaults with little tuning; native high-cardinality
categoricals (e.g. `element_id` across thousands of channels); overfitting
resistance when labels are scarce.
**DCS fit.** GPU-capable and distributable; robust when confirmed-fault labels
are few.

### 6. Logistic Regression — `algorithms/supervised/logistic_regression.py`
**What it does.** A linear decision boundary in engineered-feature space; score
= positive-class probability. `class_weight="balanced"` offsets imbalance.
**Good at.** The interpretable *floor*: near-zero cost, transparent coefficients
("which sensor drives the alarm"). Any complex model must beat it to justify its
cost.
**DCS fit.** Trivially distributed (Spark MLlib). Linear boundary only → weaker
on nonlinear fault signatures.

### 6–7. Support Vector Machine — `algorithms/supervised/svm.py`
**What it does.** An RBF-kernel SVM finds the maximum-margin boundary in an
implicit high-dimensional space; score = signed distance to the margin. Trained
on a stratified subsample because kernel SVM is `O(n²)–O(n³)`.
**Good at.** Nonlinear signatures on moderate-sized data; margin-based scoring is
naturally calibrated for thresholding.
**DCS fit.** Poor scaling and no native feature importance → included for
methodological completeness. For large-scale linear use, swap in
`SGDClassifier(loss="hinge")`.

---

## Evaluation notes

- **Threshold** is chosen by sweeping score quantiles for the best raw
  point-wise F1 — an *optimistic but uniform* convention (identical for every
  model, so the comparison is fair). Production must instead calibrate the
  threshold on healthy data via Extreme-Value-Theory / POT.
- **PA-F1** (point-adjusted F1) answers "did an alarm fire *during* the
  incident?"; raw F1 answers "how precisely was it localised?". Report both.
- **PR-AUC** is the metric to trust at a ~5 % base rate — ROC-AUC is optimistic
  under class imbalance.

## Which of these are actually used at ATLAS / CERN?

Grounded in published ATLAS/CERN work — a useful reality check on where each
method sits between "deployed" and "candidate from the literature":

**Documented in real ATLAS anomaly-detection systems**

- **DBSCAN** — the statistical branch of **DeepHYDRA** (ACM ICS 2024), a hybrid
  DBSCAN + deep-learning detector whose *projected deployment target is the
  ATLAS High-Level Trigger (HLT/TDAQ)*: DBSCAN runs on CPU for point anomalies
  while the deep model runs on GPU for long-term outliers.
- **Transformer autoencoder** — the deep branch of DeepHYDRA (the GPU-side
  reduction/reconstruction model in that same ATLAS-targeted system).
- **LSTM autoencoder** — used for **ATLAS Liquid Argon (LAr) calorimeter data
  quality monitoring**: an unsupervised LSTM encoder/decoder flags anomalous
  time periods by reconstruction loss (arXiv 2512.05977).
- **Autoencoder (general)** — the workhorse of ML-based **Data Quality
  Monitoring (DQM)** across ATLAS/CMS; reconstruction error on
  stable-operation data as an early-warning signal.
- **Isolation Forest** — appears in **hybrid autoencoder + Isolation Forest**
  DQM pipelines (train an AE on stable data, score with IF) and is widely used
  for CERN infrastructure/accelerator monitoring.

**Used elsewhere in ATLAS, but for physics rather than DCS**

- **Gradient-boosted trees (XGBoost-class BDTs)** are ubiquitous in ATLAS
  *physics analysis* (signal/background event classification). For DCS anomaly
  detection they are less common because confirmed-fault **labels are scarce**
  — which is exactly why the unsupervised family dominates operational DCS work.

**Candidate / benchmark methods (in surveys & the project proposal, not
documented ATLAS production)**

- **USAD, OmniAnomaly, VAE, GAN** — strong industrial-benchmark detectors
  (USAD/OmniAnomaly come from Orange and Alibaba respectively). They are
  referenced as candidates for a DeepHYDRA-style hybrid and are worth
  evaluating, but there is no public evidence of them running in ATLAS
  production. **K-Means / GMM / PCA** are classic baselines rather than deployed
  ATLAS detectors.

**Takeaway.** The real ATLAS direction is a **hybrid** — a fast statistical
per-point method (DBSCAN / Isolation Forest) fused with a deep temporal
reconstruction model (LSTM / Transformer autoencoder) — which is exactly the
architecture this prototype's results argue for below. Sources are listed in
the README.

## The single strongest empirical finding

Window-aggregating deep models dilute 1–2-step spikes (a 2-step spike is
2/window of the mean error and vanishes), while per-timestep methods
(Isolation Forest, DBSCAN) and last-step temporal models (LSTM, OmniAnomaly)
catch them. Conversely, spatial methods are blind to contextual/collective
faults that sequence models catch. **This is the case for a hybrid**: keep a
cheap per-timestep screen (Isolation Forest / DBSCAN) *and* a temporal deep
branch (LSTM/OmniAnomaly-style), fused under an adaptive threshold.
