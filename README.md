# Deep Learning-Based Virtual Race Engineer for Real-Time Formula 1 Strategy Optimization

A supervised multi-task deep learning system that learns from real Formula 1
telemetry to generate race-strategy recommendations — pit-stop timing, tyre
compound selection, degradation trends and race-outcome probability.

> **No reinforcement learning is used.** The entire system is supervised and
> trained on real historical telemetry accessed through the
> [FastF1](https://docs.fastf1.dev/) API, which distinguishes it from the
> simulator-based RL work in the literature.

**Course:** Deep Learning (VIT, 7th Semester)

**Team:** Dhruv Deepak (23BCE9187) · Akhil V (23BCE8051) ·
Mohan Simha Varma Dantuluri (23BCE9229) · Surya Prathap Mamillapalli (23BCE7783)

---

## The modelling task

All three models solve the **same** classification problem, so their accuracy
scores and confusion matrices are directly comparable:

> Given the last **10 laps** of telemetry for a driver, will that driver
> **pit within the next 3 laps?**

### Why a 3-lap horizon

A driver pits only 2–3 times in a ~55-lap race, so "pits on this exact lap" is
roughly **5% positive**. A model that always predicts *no pit* would score 95%
accuracy while being completely useless. Widening the target to a 3-lap window
raises the positive rate to **8.9%** (measured across all 90 races), which is
trainable but still heavily imbalanced. We therefore also up-weight the positive
class in the loss and report **precision, recall, F1 and PR-AUC** alongside raw
accuracy, because accuracy alone hides this failure mode.

## The three models

| # | Model | Role | Reference |
|---|-------|------|-----------|
| 1 | **Bi-LSTM** | Sequential baseline | Sasikumar et al., 2025 |
| 2 | **CNN–BiLSTM** | Convolutional feature extraction + sequence model | Sasikumar et al., 2025 |
| 3 | **Multi-task Transformer with cross-driver attention** | Proposed architecture; trained on four heads, evaluated on its pit head | This work |

Model 3 is the project's contribution: a shared representation across four
strategic tasks, with an attention mechanism over all drivers on track so that
competitor behaviour directly influences each prediction.

**"Most efficient"** is reported on two axes:

* **Predictive** — accuracy, precision, recall, F1, PR-AUC, confusion matrix
* **Computational** — parameter count, training time, per-lap inference latency
  (the system claims *real-time*, so this matters)

---

## Dataset

| Property | Value |
|---|---|
| Source | FastF1 API (official F1 timing + telemetry + Ergast archive) |
| Seasons | 2021 – 2024 |
| Sessions | Race sessions |
| Scale | 90 races, 98,357 driver-lap rows, 81,270 sequence windows |

Raw car telemetry is sampled at ~3.7 Hz, which would be tens of gigabytes across
all sessions. It is therefore **aggregated per lap at download time** (mean/max
speed, throttle-on %, brake %, average gear and RPM) rather than stored raw.

The FastF1 API allows **500 calls per hour** and one race costs roughly nine of
them, so a full sweep cannot finish inside a single window. `download.py` sleeps
through the rate limit and resumes, and skips races already on disk, so it can be
re-run freely.

### Splits

Split by **season**, never by random row — two laps from the same race are
highly correlated, so a random split leaks the test set into training.

| Split | Seasons | Races | Windows | Positives |
|---|---|---|---|---|
| Train | 2021, 2022 | 43 | 38,890 | 9.15% |
| Validation | 2023 | 22 | 20,152 | 9.55% |
| Test | 2024 | 24 | 22,228 | 8.52% |

---

## Project structure

```
f1-virtual-race-engineer/
├── src/
│   ├── config.py           # paths, splits, hyper-parameters, seeding
│   ├── data/               # FastF1 download and caching
│   ├── features/           # feature engineering, labels, sequence windows
│   ├── models/             # the three architectures
│   ├── evaluation/         # metrics, curves, confusion matrices
│   └── utils/
├── notebooks/              # review presentations (import from src/, no logic of their own)
│   ├── 01_review2_bilstm.ipynb                 Model 1 end to end
│   ├── 02_review3_model2_cnn_bilstm.ipynb      Model 2
│   ├── 03_review3_model3_attention.ipynb       Model 3 + the attention ablation
│   └── 04_review3_comparison_and_replay.ipynb  three-model comparison + race replay
├── data/                   # raw + processed data (gitignored, regenerable)
├── models/                 # trained weights (gitignored)
├── reports/
│   ├── figures/            # accuracy/loss curves, confusion matrices
│   └── metrics/            # results tables
└── cache/                  # FastF1 cache (gitignored, grows to several GB)
```

The pipeline lives in `src/` as plain Python modules so that all three models
import identical preprocessing code. Notebooks import from `src/` and exist to
present results, not to hold logic.

---

## Notebooks

Four executed notebooks walk through the work with live outputs and every figure embedded. They
import from `src/` rather than redefining anything, so they cannot drift from the committed
pipeline.

| Notebook | Covers |
|---|---|
| `01_review2_bilstm.ipynb` | Model 1 end to end — the Review 2 presentation |
| `02_review3_model2_cnn_bilstm.ipynb` | Model 2, and the sweep that selected its widths |
| `03_review3_model3_attention.ipynb` | Model 3, its masking tests, and the attention ablation |
| `04_review3_comparison_and_replay.ipynb` | All three compared, classical baselines, race replay |

Notebooks 02–04 default to `RETRAIN = False`: they load the weights and predictions the committed
training runs saved, so their numbers match the report exactly. Set `RETRAIN = True` in a notebook
to train a fresh copy live — it fits the model inside the notebook and does not overwrite the
published artefacts.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Training runs on CPU: the pinned TensorFlow build is not
compiled with CUDA, and on native Windows TensorFlow has not shipped GPU support
since 2.10. Model 1 trains in well under two minutes regardless, so this has not
been worth changing.

```bash
# Phase 1 — download and cache the FastF1 data (slow on first run, ~30-60 min)
python -m src.data.download

# Phase 2 — build the model-ready dataset
python -m src.features.build_dataset

# Phase 3 — train a model
python -m src.models.train --model bilstm
```

---

## Development phases

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Repository scaffolding and configuration | ✅ |
| 1 | FastF1 data acquisition (2021–2024) | ✅ |
| 2 | Feature engineering and sequence dataset | ✅ |
| 3 | **Model 1: Bi-LSTM** — training, curves, confusion matrix | ✅ |
| 4 | Model 2: CNN–BiLSTM | ✅ |
| 5 | Model 3: Multi-task Transformer with cross-driver attention | ✅ |
| 6 | Three-model comparison + historical race replay | ✅ |
| 7 | Final report and presentation | ⬜ |

---

## Results so far

Test set: the full 2024 season, 22,228 windows, 8.52% positive.

| Model | Params | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Never-pit baseline | — | 0.9148 | 0.5000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.085 |
| Logistic regression (final lap) | — | — | — | — | — | — | 0.802 | 0.308 |
| Gradient boosting (flat window) | — | — | — | — | — | — | **0.853** | **0.391** |
| **Model 1 — Bi-LSTM** @ 0.50 | 104,769 | 0.7285 | **0.7271** | 0.199 | **0.725** | 0.313 | **0.806** | **0.323** |
| **Model 1 — Bi-LSTM** @ tuned 0.779 | 104,769 | 0.8642 | 0.6713 | 0.298 | 0.439 | **0.355** | 0.806 | 0.323 |
| **Model 2 — CNN–BiLSTM** @ tuned 0.796 | 8,881 | 0.8404 | 0.6760 | 0.261 | 0.478 | 0.338 | 0.772 | 0.306 |
| **Model 3 — MTL + cross-driver attention** @ tuned 0.708 | 48,394 | 0.8177 | 0.6732 | 0.233 | 0.499 | 0.318 | 0.756 | 0.265 |

### Race replay — AI recommendations vs. actual team decisions

Per-lap metrics answer *"how often is the model right about a lap"*. A race
engineer asks something different: **of the stops that happened, how many did we
see coming, and how early?** The replay scores per *stop* across all 24 races of
the 2024 season — none of which any model saw in training.

Consecutive firing laps before one stop count as a single alarm, not three;
counting each lap separately would badly overstate the noise, because the label
itself spans a three-lap window.

| Model | Stops anticipated | % | Median lead | False alarms per driver-race |
|---|---|---|---|---|
| Model 1 — Bi-LSTM | 314 / 644 | 48.8% | 3 laps | **0.14** |
| Model 2 — CNN–BiLSTM | **356 / 644** | **55.3%** | 3 laps | 0.22 |
| Model 3 — MTL + attention | 355 / 644 | 55.1% | 3 laps | 0.27 |

**The operational ranking is not the per-lap ranking.** Model 1 leads on PR-AUC
but anticipates the *fewest* stops; its higher tuned threshold makes it the most
conservative, which buys the lowest false-alarm rate. Model 2 catches 42 more
stops across the season for one extra false alarm roughly every 12 driver-races.
Which is preferable is a strategy call, not a modelling one — and that the two
views disagree is exactly what the replay exists to expose.

In absolute terms the false-alarm rate is low enough to be usable: a strategist
sees a spurious warning roughly once every four to seven driver-races, while
getting about three laps of warning on more than half of all real stops.

`reports/figures/replay_grid_2024_11.png` shows the whole field for one race —
probability building across every driver's stint and collapsing the moment they
stop.

### Model 3 — Multi-task with cross-driver attention

The project's proposed contribution, and the one mechanism neither other model
has. A shared Bi-LSTM encodes each driver's window with the same weights, then
multi-head attention runs **across the driver axis** with a residual and layer
norm, followed by a position-wise feed-forward block — a transformer block
applied over drivers rather than over tokens. Four heads share the result: pit,
next compound, next-lap pace, and race outcome.

To make this possible the samples are regrouped into `(race, lap)` grids of up
to 20 drivers. The feature tensors are taken straight from `sequences.npz` and
only reindexed, so Model 3 sees byte-identical features, the identical
train-only scaler and the identical season splits as Models 1 and 2.

#### Ablation: does cross-driver attention earn its place?

A width sweep would say which Model 3 is best; it would not answer the question
the project is actually asking. So the sweep is a 2×2 ablation — attention
on/off crossed with multi-task on/off — plus two width variants. Selection on
validation PR-AUC, as for the other models. Full results in
[`reports/metrics/mtl_ablation.csv`](reports/metrics/mtl_ablation.csv),
reproducible via `python scripts/ablate_model3.py`.

| Config | Attention | Multi-task | Params | Val PR-AUC | Test PR-AUC |
|---|---|---|---|---|---|
| A — full Model 3 | ✅ | ✅ | 48,394 | 0.2936 | 0.265 |
| B | ✅ | — | 48,394 | 0.3007 | 0.280 |
| C | — | ✅ | 31,754 | **0.3122** | **0.312** |
| D — control | — | — | 31,754 | 0.3112 | 0.308 |

| Effect | Comparison | Δ Test PR-AUC |
|---|---|---|
| Attention, with multi-task | A vs C | **−0.047** |
| Attention, pit head only | B vs D | **−0.027** |
| Multi-task, with attention | A vs B | −0.015 |
| Multi-task, no attention | C vs D | +0.004 |

**The finding is negative and consistent: cross-driver attention hurts.** It
costs performance in both conditions, with and without the auxiliary heads.
Multi-task learning is roughly neutral — slightly negative alongside attention,
slightly positive without it.

Two explanations, neither flattering but both testable:

* **The grid layout collapses the training signal.** Regrouping by `(race, lap)`
  turns 38,890 training windows into **2,181 training grids**. The same
  driver-laps are covered, but the number of gradient updates per epoch falls
  roughly 18-fold — and attention adds parameters at precisely the moment the
  number of examples drops.
* **The competitor context is already in the features.** Gap to leader,
  intervals ahead and behind, share of the field already stopped, and tyre age
  relative to the field are all engineered into the 55 inputs. Attention may be
  relearning, noisily and from far fewer examples, what those features already
  state directly.

Model 3's reported row is configuration A, the architecture as proposed.
Configurations C and D are ablation *controls*, not candidate configurations:
switching attention off changes what the model is, unlike the width sweeps that
selected Models 1 and 2. Reporting C as "Model 3" would mislabel the
architecture and bury the result.

### Model 2 — CNN–BiLSTM

Two `Conv1D` layers with batch normalisation feed a bidirectional LSTM. The
convolutions slide over the *lap* axis to pick up short local motifs — a
three-lap burst of degradation, a lap-time spike from traffic or a safety car —
leaving the recurrent layer to model only how those evolve. There is no pooling
layer: at `SEQ_LEN = 10` even a stride-2 pool would discard half the temporal
resolution the recurrent layer exists to read.

Widths come from a six-configuration sweep matching the one Model 1 received,
selected on validation PR-AUC alone. **Sixteen filters with a 16-unit LSTM —
8,881 parameters, a twelfth of Model 1 — generalised best**, while the widest
variant (64 filters, kernel 5) came last. Widening the kernel hurt most: at a
10-lap window a kernel of 5 spans half the sequence, so the convolution stops
detecting local motifs and starts duplicating the recurrent layer.

Every configuration in the sweep landed between 0.259 and 0.301 test PR-AUC, all
below Model 1. **Model 2 trailing Model 1 is a property of the architecture on
this data, not of how hard each was tuned.** It is, however, 12× smaller and
slightly faster per lap at inference (0.048 ms vs 0.061 ms), which is the
efficiency half of the Review 3 comparison.

Note the never-pit baseline: it wins on raw accuracy and scores zero on every
metric that matters. That is exactly why accuracy is not the headline number here.

### Honest reading of this result

Gradient boosting on the flattened window currently **outperforms** the Bi-LSTM,
and logistic regression on the final lap alone roughly matches it. The sequence
model is not yet earning its complexity on this task. Two reasons, both worth
stating:

* **Effective sample size.** 38,890 training windows come from only 815
  driver-races across 43 races, and consecutive windows share 9 of their 10 laps.
  The nominal sample count badly overstates how much independent data there is.
* **Capacity is not the bottleneck.** A sweep over six architectures from 104k
  down to 10k parameters, varying dropout, L2 and recurrent dropout, moved
  validation PR-AUC only between 0.288 and 0.320. The model overfits within one
  to three epochs regardless.

This sets a real bar for Models 2 and 3: the multi-task model with cross-driver
attention has to beat 0.391 PR-AUC, not just the trivial baseline.

A further note on thresholds: F1-optimal tuning pushes the threshold to 0.779,
trading recall (0.725 → 0.439) for precision. For an actual race engineer that
is likely the wrong trade — a missed stop is expensive while a false alarm costs
nothing — so the 0.50 operating point, with 72.5% of stops caught, is arguably
the more useful one. Both are reported rather than silently choosing.

---

## References

1. FastF1 Documentation — https://docs.fastf1.dev/
2. Sasikumar, A., Leema, A. A., & Balakrishnan, P. (2025). *Data-driven pit stop
   decision support for Formula 1 using deep learning models.* Frontiers in
   Artificial Intelligence, 8, 1673148.
3. Thomas, D., et al. (2025). *Explainable reinforcement learning for Formula One
   race strategy.* ACM/SIGAPP SAC.
4. Veerasagar, S. S., Coelho, R., & Chandrashekar, L. K. (2025). *Optimum racing:
   A F1 strategy predictor using reinforcement learning.* IJRASET.
5. Vaswani, A., et al. (2017). *Attention is all you need.* NeurIPS 30.
