"""
Is the sweep's winner actually better, or is it noise?

The fair-batch sweep spread only 0.026 validation PR-AUC across eight
configurations, and its validation and test rankings disagree -- the config that
won validation placed third on test. Both are signs that single-seed differences
at this resolution are not distinguishable from run-to-run variance.

This runs the three contenders across three seeds each and reports mean and
spread, so the decision to keep or change Model 3's published configuration
rests on evidence rather than on one run.

    A  batch 32, 48,394 params   -- the published configuration
    F  batch 14, 162,314 params  -- won the sweep on validation
    G  batch 14, 13,914 params   -- nearly matched F on test at a twelfth the size

Usage
-----
    python scripts/seed_check_model3.py
"""

from __future__ import annotations

import time
import warnings

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

from src import config
from src.models.mtl_architecture import build_mtl_attention
from src.models.train_mtl import grid_arrays, load_grids, scatter_to_rows

SEEDS = [42, 1337, 2024]
CONFIGS = [
    ("A published  48k", dict(batch=32)),
    ("F sweep best 162k", dict(batch=14, encoder_units=64, attn_key_dim=32, ff_units=128)),
    ("G narrow      14k", dict(batch=14, encoder_units=16, attn_heads=2,
                               attn_key_dim=8, ff_units=32)),
]

seq, mtl, spec, X, n_train, n_val = load_grids()
n_rows = len(X)
x_tr, y_tr, _, _ = grid_arrays(mtl, X, "train", spec)
x_va, y_va, g_va, m_va = grid_arrays(mtl, X, "val", spec)
x_te, y_te, g_te, m_te = grid_arrays(mtl, X, "test", spec)
pit_tr = y_tr["pit"][x_tr["driver_mask"] > 0]
pw = float((len(pit_tr) - pit_tr.sum()) / pit_tr.sum())
y_test = seq["y_test"]
test_slice = slice(n_train + n_val, n_rows)

rows = []
for tag, kw in CONFIGS:
    kw = dict(kw)
    batch = kw.pop("batch")
    for seed in SEEDS:
        config.set_seed(seed)
        m = build_mtl_attention(seq_len=spec["seq_len"], n_features=spec["n_features"],
                                max_drivers=spec["max_drivers"],
                                n_compounds=spec["n_compound_classes"],
                                n_outcomes=spec["n_outcome_classes"],
                                pos_weight=pw, learning_rate=config.LEARNING_RATE, **kw)
        t0 = time.time()
        h = m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=40, batch_size=batch,
                  verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                                           patience=8, restore_best_weights=True)])
        te = scatter_to_rows(m.predict(x_te, batch_size=64, verbose=0)["pit"][..., 0],
                             g_te, m_te, n_rows)[test_slice]
        rows.append({"config": tag, "seed": seed, "params": m.count_params(),
                     "val_pr_auc": round(max(h.history["val_pr_auc"]), 4),
                     "test_pr_auc": round(float(average_precision_score(y_test, te)), 4)})
        print(f"{tag:<20} seed {seed:<5} val {rows[-1]['val_pr_auc']:.4f}  "
              f"test {rows[-1]['test_pr_auc']:.4f}  {time.time()-t0:>5.0f}s")

df = pd.DataFrame(rows)
out = config.METRICS_DIR / "mtl_seed_check.csv"
df.to_csv(out, index=False)

print("\nacross seeds (mean +/- spread):")
agg = df.groupby("config").agg(
    params=("params", "first"),
    val_mean=("val_pr_auc", "mean"), val_std=("val_pr_auc", "std"),
    test_mean=("test_pr_auc", "mean"), test_std=("test_pr_auc", "std")).round(4)
print(agg.to_string())

spread = df.groupby("config")["test_pr_auc"].agg(lambda s: s.max() - s.min())
print(f"\nwidest seed-to-seed test swing within one config: {spread.max():.4f}")
print(f"gap between best and worst config mean          : "
      f"{agg['test_mean'].max() - agg['test_mean'].min():.4f}")
print(f"\nModel 2 reference test PR-AUC: 0.3055")
print("Model 1 reference test PR-AUC: 0.3215")
print(f"\nsaved {out}")
