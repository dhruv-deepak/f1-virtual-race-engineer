"""
Give Model 3 a fair optimisation budget.

The problem this addresses
--------------------------
Regrouping samples into (race, lap) grids changed more than the architecture: it
changed the optimisation regime. Models 1 and 2 take 38,890 windows at batch 256
-- 152 gradient steps per epoch, 256 driver-laps per step. Model 3 took 2,181
grids at batch 32 -- only 69 steps per epoch, each covering ~570 driver-laps.

With every model stopped by the same early-stopping patience, Model 3 was
getting less than half the gradient updates before training halted. Any
comparison against Models 1 and 2 therefore confounded the architecture with the
optimisation budget.

Batch 14 restores parity on both axes at once: 2,181 / 14 = 156 steps per epoch
(against 152), and 14 x 17.8 drivers = ~249 driver-laps per step (against 256).

Selection is on validation PR-AUC, as for every other model in the project. Test
PR-AUC is printed alongside but plays no part in choosing.

Usage
-----
    python scripts/fair_batch_model3.py
"""

from __future__ import annotations

import time
import warnings

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

from src import config
from src.models.mtl_architecture import build_mtl_attention
from src.models.train_mtl import grid_arrays, load_grids, scatter_to_rows

seq, mtl, spec, X, n_train, n_val = load_grids()
n_rows = len(X)
x_tr, y_tr, _, _ = grid_arrays(mtl, X, "train", spec)
x_va, y_va, g_va, m_va = grid_arrays(mtl, X, "val", spec)
x_te, y_te, g_te, m_te = grid_arrays(mtl, X, "test", spec)

pit_tr = y_tr["pit"][x_tr["driver_mask"] > 0]
pw = float((len(pit_tr) - pit_tr.sum()) / pit_tr.sum())
y_val, y_test = seq["y_val"], seq["y_test"]
val_slice = slice(n_train, n_train + n_val)
test_slice = slice(n_train + n_val, n_rows)

n_grids = len(x_tr["windows"])
print(f"train grids {n_grids:,}  pos_weight {pw:.2f}")
print(f"Models 1-2 reference: 152 steps/epoch, 256 driver-laps per step\n")

CONFIGS = [
    ("A  batch 32  (as published)", dict(batch=32)),
    ("B  batch 14  (step-matched)", dict(batch=14)),
    ("C  batch  8", dict(batch=8)),
    ("D  batch  4", dict(batch=4)),
    ("E  batch 14, lr 1e-3", dict(batch=14, lr=1e-3)),
    ("F  batch 14, wider", dict(batch=14, encoder_units=64, attn_key_dim=32, ff_units=128)),
    ("G  batch 14, narrower", dict(batch=14, encoder_units=16, attn_heads=2,
                                   attn_key_dim=8, ff_units=32)),
    ("H  batch  8, dropout 0.3", dict(batch=8, dropout=0.3)),
]

rows = []
for tag, kw in CONFIGS:
    batch = kw.pop("batch")
    lr = kw.pop("lr", config.LEARNING_RATE)
    config.set_seed()
    m = build_mtl_attention(seq_len=spec["seq_len"], n_features=spec["n_features"],
                            max_drivers=spec["max_drivers"],
                            n_compounds=spec["n_compound_classes"],
                            n_outcomes=spec["n_outcome_classes"],
                            pos_weight=pw, learning_rate=lr, **kw)
    t0 = time.time()
    h = m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=40, batch_size=batch, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                                       patience=8, restore_best_weights=True)])
    vp = h.history["val_pr_auc"]
    te = scatter_to_rows(m.predict(x_te, batch_size=64, verbose=0)["pit"][..., 0],
                         g_te, m_te, n_rows)[test_slice]
    te_pr = average_precision_score(y_test, te)
    te_roc = roc_auc_score(y_test, te)
    steps = int(np.ceil(n_grids / batch))
    rows.append({"config": tag, "batch": batch, "steps_per_epoch": steps,
                 "params": m.count_params(), "epochs_run": len(vp),
                 "best_epoch": int(np.argmax(vp)) + 1, "val_pr_auc": round(max(vp), 4),
                 "test_pr_auc": round(float(te_pr), 4), "test_roc_auc": round(float(te_roc), 4),
                 "train_s": round(time.time() - t0, 1)})
    print(f"{tag:<30} steps/ep {steps:>4}  val PR {max(vp):.4f} @ep{int(np.argmax(vp))+1:<2} "
          f"of {len(vp):<2}  test PR {te_pr:.4f} ROC {te_roc:.4f}  {time.time()-t0:>5.0f}s")

df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
out = config.METRICS_DIR / "mtl_fair_batch_sweep.csv"
df.to_csv(out, index=False)

print("\nranked by validation PR-AUC (the selection metric):")
print(df[["config", "batch", "steps_per_epoch", "val_pr_auc", "test_pr_auc", "test_roc_auc"]]
      .to_string(index=False))

published = next(r for r in rows if r["config"].startswith("A"))
best = df.iloc[0]
print(f"\npublished (batch 32): val {published['val_pr_auc']:.4f}  test {published['test_pr_auc']:.4f}")
print(f"best in sweep       : val {best['val_pr_auc']:.4f}  test {best['test_pr_auc']:.4f}  "
      f"({best['config'].strip()})")
print(f"delta on test       : {best['test_pr_auc'] - published['test_pr_auc']:+.4f}")
print(f"\nModel 1 reference   : test PR-AUC 0.3232")
print(f"Model 2 reference   : test PR-AUC 0.3055")
print(f"\nsaved {out}")
