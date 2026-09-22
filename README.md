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
├── notebooks/              # review presentations (import from src/)
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

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA GPU is optional but speeds up training
considerably (developed on an RTX 3050).

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
| 3 | **Model 1: Bi-LSTM** — training, curves, confusion matrix (*Review 2*) | ✅ |
| 4 | Model 2: CNN–BiLSTM | ⬜ |
| 5 | Model 3: Multi-task Transformer with cross-driver attention | ⬜ |
| 6 | Three-model comparison + historical race replay (*Review 3*) | ⬜ |
| 7 | Final report and presentation | ⬜ |

---

## Results so far — Model 1 (Bi-LSTM)

Test set: the full 2024 season, 22,228 windows, 8.52% positive.

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Never-pit baseline | 0.9148 | 0.5000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.085 |
| Logistic regression (final lap) | — | — | — | — | — | 0.802 | 0.308 |
| Gradient boosting (flat window) | — | — | — | — | — | **0.853** | **0.391** |
| **Bi-LSTM @ 0.50** | 0.7285 | **0.7271** | 0.199 | **0.725** | 0.313 | 0.806 | 0.323 |
| **Bi-LSTM @ tuned (0.779)** | 0.8642 | 0.6713 | 0.298 | 0.439 | **0.355** | 0.806 | 0.323 |

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
