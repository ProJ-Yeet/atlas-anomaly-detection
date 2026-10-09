# Graphs Guide — How to Read Every Figure

Every figure this project produces lives in `atlas_anomaly_detection/figures/`.
There are three kinds:

1. **Per-model diagnostics** — one `<model>.png` per algorithm (a 3×3 grid).
2. **Cross-model analysis** — `comparison.png`, `anomaly_type_coverage.png`,
   `best_algo_per_type.png`, `model_complementarity.png`,
   `ensemble_greedy_coverage.png`.
3. **Standalone evaluation plots** — `<model>_roc.png`, `<model>_pr.png`,
   `<model>_cm.png` (produced on demand by the `evaluation/` scripts).

Regenerate: `python main.py` (per-model + comparison), then
`python analyze_results.py` (the four analysis graphs).

**One convention everywhere:** the *anomaly score* is always oriented so that
**higher = more anomalous**, and a single **threshold** turns the score into a
0/1 prediction. Blue = normal, red = anomaly, throughout.

### What is "overall performance" vs what is an "illustrative example"

This trips people up, so read this first:

- **Every performance number uses ALL the data — not a sample.** The score
  histogram, ROC, PR, confusion matrix, `comparison.png`,
  `anomaly_type_coverage.png` and the CSVs are computed over *every row each
  model scored*: **100,000 rows** for the unsupervised models and the
  **40,000-row held-out test split** for the supervised models. These are the
  overall-performance panels.
- **All 18 models run on the same dataset.** There is one generated dataset;
  nothing is trained or evaluated on different data. (The supervised models see
  fewer rows only because they are *correctly* scored on a held-out test split —
  standard train/test methodology — while unsupervised models, having no labels
  and thus no train/test split, are scored on everything.)
- **Only two panels are illustrative examples, by design:**
  1. **The time-series panel (Panel 1–2)** plots **one element over one time
     window**, and the **`temperature` channel specifically**, purely so a fault
     is legible on a readable axis — you cannot draw 100k timesteps across 40
     channels on one line plot. **`temperature` is only the *display* channel;
     the model actually uses all 41 features.** As of the latest version **every
     model's time-series panel shows the *same* reference element and window**
     (the element whose test window contains the most anomalies), so they are now
     directly comparable — earlier versions picked each model's own worst element,
     which made the panels look like "different data".
  2. **The PCA scatter (Panel 7–8)** draws a **deterministic 6,000-point sample**
     (same seed for every model, all anomalies always kept) — only because a
     100k-point scatter is an unreadable, slow-to-render blob. The *decision* it
     illustrates was still made on all the data.

So: to judge **overall performance**, read Panels 3–6 and the analysis graphs.
Panels 1–2 answer "what does one incident look like and did the model score it?"
and Panels 7–8 answer "what does the decision boundary look like in 2-D?".

---

## 1. The per-model 9-panel figure — `<model>.png`

The main diagnostic for one algorithm (e.g. `gmm.png`, `omnianomaly.png`). A
3×3 grid. The **top-left two panels show a single reference element and the
temperature display channel** (the same element for every model, so figures are
comparable); **all other panels use the full dataset each model scored**.

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ 1  time series      │ 2  anomaly score    │ 3  score histogram  │
│    (one element)    │    + threshold      │    by class         │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ 4  ROC curve        │ 5  PR curve         │ 6  confusion matrix │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ 7  PCA (predicted)  │ 8  PCA (truth)      │ 9  feature importance│
└─────────────────────┴─────────────────────┴─────────────────────┘
```

**Panel 1 — Time series with anomaly overlay.** A blue line of the temperature
**display channel** over the reference element's time window; **red shaded
bands** mark the *ground-truth* anomalous periods. The same element is shown for
every model. Purpose: see what a real fault looks like in the raw signal.
*Read it:* do the red bands coincide with visibly unusual behaviour? Some faults
(contextual, collective, or ones that only show in *other* channels like voltage
or heartbeat) look normal on the temperature line on purpose — that is exactly
why the model needs all 41 features, not just temperature.

**Panel 2 — Anomaly score + threshold + flags.** The model's score for the same
element window (grey line), a **red dashed horizontal threshold**, and **red
dots** where the score crosses it (the model's alarms). *Read it:* the score
should spike inside the Panel-1 red bands. Red dots inside a true band = correct
detection; red dots outside = false alarms. Because every model shares this
element, you can compare Panel 2 across two models' figures directly to see which
localises the same incident better.

**Panel 3 — Score histogram by class (log y-axis).** Distribution of scores for
**normal** rows (blue) vs **anomalous** rows (red), with the threshold as a
vertical line. *Read it:* good separation = the red mass sits to the right of
the blue mass with little overlap. The y-axis is **log-scaled** because normals
outnumber anomalies ~20:1 — without it the red bars would be invisible. The
threshold line shows where the model cut; anything to its right is flagged.

**Panel 4 — ROC curve.** True-positive rate vs false-positive rate as the
threshold sweeps; the grey diagonal is random guessing; the legend shows
**AUC**. *Read it:* closer to the top-left corner is better; AUC 1.0 = perfect,
0.5 = random. **Caution:** at a ~5 % anomaly rate ROC-AUC is *optimistic*
(dominated by the abundant negatives) — treat it as a secondary metric and trust
Panel 5.

**Panel 5 — Precision-Recall curve.** Precision vs recall as the threshold
sweeps; the grey dashed line is the **base rate** (~0.05, the PR-AUC of random
guessing); the legend shows **AP / PR-AUC**. *Read it:* this is the **metric to
trust** on this imbalanced problem. The curve should sit far above the base-rate
line; the further toward the top-right, the better. A curve hugging the base
rate means the model barely beats chance at finding the rare class.

**Panel 6 — Confusion matrix.** Counts at the model's chosen threshold, Blues
shaded. Layout: rows = truth, columns = prediction, so
top-left = **TN**, top-right = **FP** (false alarms), bottom-left = **FN**
(missed anomalies), bottom-right = **TP** (caught). *Read it:* you want a large
TN, large-ish TP, and small FP/FN. For DCS, FN (missed faults) is usually the
costly cell; FP (nuisance alarms) is the annoyance cell.

**Panel 7 — PCA scatter coloured by *prediction*.** The 41-D feature space
projected to 2-D via PCA (subsampled to ~6 000 points, all anomalies kept),
coloured by what the model **predicted** (0 vs 1). *Read it:* shows the model's
decision *geometry* — where in feature space it draws the anomaly region.

**Panel 8 — PCA scatter coloured by *truth*.** The same 2-D projection coloured
by the **ground-truth** label. *Read it:* compare Panel 7 vs Panel 8 side by
side — where they agree the model is right; where Panel 7 colours differ from
Panel 8 are the errors. **Two cautions:** (1) PCA keeps only ~2 directions of
variance, so points that overlap here may be perfectly separable in the full
41-D space — overlap is not proof of an impossible problem. (2) The anomaly
class (orange, label 1) is drawn *on top* of the normal class (blue), so where
both share a region the panel looks orange-heavy — it is **not** saying most
rows are anomalous (the confusion matrix shows the true ~20:1 balance). It
highlights where anomalies sit on the healthy manifold, which is the point.

**Panel 9 — Feature importance (supervised models only).** Horizontal green bars
of the top-15 most influential features. For unsupervised models this panel is
blank (they have no supervised importances). *Read it:* which engineered signals
drive the decision (e.g. `temperature_roc`, `heartbeat_delta`,
`current_roll_std`). Sanity-checks the model against physics.

---

## 2. Cross-model analysis graphs

### `comparison.png` — the leaderboard
Grouped bar chart: for every model, four bars — **precision, recall, F1,
PR-AUC** (y-axis 0–1). *Read it:* tall + balanced bars = a strong all-rounder.
A tall precision bar with a short recall bar = a cautious model (few false
alarms, but misses faults); the reverse = a trigger-happy model. Use PR-AUC as
the single fairest column at this base rate. Source table:
`figures/comparison.csv`.

### `anomaly_type_coverage.png` — who catches which fault type
Heatmap. **Rows = models** (plus an `alarm_baseline` row); **columns = the 11
anomaly types**; **cell = recall** (fraction of that fault type the model
flagged), 0→1 on a viridis scale (dark = missed, bright = caught), with the
number printed in each cell. *Read it:*
- Scan a **column** to find the best model(s) for one fault type.
- Scan a **row** to see one model's strengths/blind spots across fault types.
- The **`alarm_baseline` row** is the naive threshold detector — where it is dark
  (e.g. `sensor_freeze`, `collective_anomaly`, `contextual_anomaly` = 0.00) but a
  model row is bright, that is **ML catching what threshold alarms miss** — the
  core value story.
- **Caveat:** supervised rows are `NaN`/absent for `collective`, `cooling`,
  `cascading` (those faults fell in the training portion of the temporal split,
  not the test rows supervised models are scored on). Rare types
  (`point_anomaly`, ~6 rows) have noisy recall — read them as "caught / missed",
  not precise percentages. Source: `figures/anomaly_type_coverage.csv`.

### `best_algo_per_type.png` — the per-type champion
Bar chart: **x = anomaly type**, **bar height = the best coverage achieved**, and
the **winning model's name printed on each bar**. *Read it:* a one-glance summary
of "use model X for fault Y". Short bars (contextual, persistent) mark the
**hard faults no model handles well** — the research frontier.

### `model_complementarity.png` — which models are redundant vs complementary
Heatmap, **models × models**, cell = correlation of their *catch vectors* (did
they flag the same true anomalies?), on a red→green scale where **red = highly
correlated (redundant)** and **green = low correlation (complementary)**. *Read
it:*
- **Red blocks** = models that catch the same anomalies → combining them adds
  nothing (e.g. the boosters with each other; the spatial methods with each
  other).
- **Green cells** = models that catch *different* anomalies → these are the pairs
  worth combining (typically a spatial method × a temporal method).
- This is the visual justification for building a **hybrid across families**
  rather than stacking near-duplicates.

### `ensemble_greedy_coverage.png` — how many models you actually need
Line-with-markers. **x = models in greedy-add order** (each step adds the model
that catches the most *still-missed* anomalies); **y = cumulative union recall**.
*Read it:* the curve rises steeply for the first 2–3 models then flattens. The
**"elbow"** tells you the minimal complementary set (here ≈ Random Forest → GMM →
Transformer); models added after the flattening are redundant. The height of the
plateau (~0.66) is the **coverage ceiling** — the fraction of anomalies *any*
combination can catch — capped by the faults every model misses (contextual /
persistent).

---

## 3. Standalone evaluation plots

Produced individually by the `evaluation/` scripts, e.g.
`python evaluation/roc_curve.py figures/isolation_forest_predictions.csv`:

- **`<model>_roc.png`** — the ROC curve alone (same as Panel 4), larger.
- **`<model>_pr.png`** — the PR curve alone (same as Panel 5), larger.
- **`<model>_cm.png`** — the confusion matrix alone (same as Panel 6), larger.

Use these when you want a single publication-size plot for one model rather than
the full 9-panel grid.

---

## 4. Quick "is this good?" cheat-sheet

| you see… | it means… |
|---|---|
| Panel 3: red mass clearly right of blue | scores separate the classes well ✅ |
| Panel 3: red and blue overlap heavily | model can't tell them apart ⚠️ |
| Panel 5 (PR) hugging the base-rate line | barely beats chance on the rare class ⚠️ |
| Panel 6: large FN cell | missing real faults (dangerous for DCS) ⚠️ |
| Panel 6: large FP cell | nuisance false alarms (operator fatigue) |
| High ROC-AUC but low PR-AUC | the imbalance is flattering ROC — trust PR |
| Coverage heatmap: bright model row, dark alarm row | ML beats the threshold baseline ✅ |
| Complementarity: green between two models | good candidates to ensemble together ✅ |
| Greedy curve flat after 3 models | you only need those 3; the rest are redundant |

**The two metrics to lead with on this problem:** **PR-AUC** (threshold-free
quality at a 5 % base rate) and **PA-F1** (point-adjusted F1 — did an alarm fire
*during* each incident, which is what operations cares about). Raw F1 measures
how precisely each incident was localised; ROC-AUC is the optimistic one.
