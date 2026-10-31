"""
Model 3 sweep and ablation.

Width variants alone would not answer the question the project is actually
asking, which is whether cross-driver attention earns its place. So the grid is
a 2x2 ablation -- attention on/off crossed with multi-task on/off -- plus two
width variants of the full model.

    A  attention + multi-task   the full Model 3
    B  attention + pit only     isolates what multi-task contributes
    C  no attention + multi-task  isolates what attention contributes
    D  no attention + pit only  control: a per-driver encoder, nothing shared

A vs C answers "does attention help?". A vs B answers "does multi-task help?".
Selection is on validation PR-AUC, as for Models 1 and 2.
"""
import time
import warnings

import keras
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

from src import config
from src.models.mtl_architecture import build_mtl_attention
from src.models.train_mtl import grid_arrays, load_grids, scatter_to_rows

PIT_ONLY = {"pit": 1.0, "compound": 0.0, "laptime": 0.0, "outcome": 0.0}

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
print(f"pos_weight={pw:.2f}  val base={y_val.mean():.4f}  test base={y_test.mean():.4f}\n")

CONFIGS = [
    ("A  attn + multi-task  (full Model 3)", dict(use_attention=True)),
    ("B  attn + pit only",                   dict(use_attention=True, task_weights=PIT_ONLY)),
    ("C  no attn + multi-task",              dict(use_attention=False)),
    ("D  no attn + pit only  (control)",     dict(use_attention=False, task_weights=PIT_ONLY)),
    ("E  attn + multi-task, wider",          dict(use_attention=True, encoder_units=64,
                                                  attn_key_dim=32, ff_units=128)),
    ("F  attn + multi-task, narrower",       dict(use_attention=True, encoder_units=16,
                                                  attn_heads=2, attn_key_dim=8, ff_units=32)),
]

rows = []
for tag, kw in CONFIGS:
    config.set_seed()
    m = build_mtl_attention(seq_len=spec["seq_len"], n_features=spec["n_features"],
                            max_drivers=spec["max_drivers"],
                            n_compounds=spec["n_compound_classes"],
                            n_outcomes=spec["n_outcome_classes"],
                            pos_weight=pw, learning_rate=config.LEARNING_RATE, **kw)
    t0 = time.time()
    h = m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=20, batch_size=32, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                                       patience=6, restore_best_weights=True)])
    vp = h.history["val_pr_auc"]
    te = scatter_to_rows(m.predict(x_te, batch_size=64, verbose=0)["pit"][..., 0],
                         g_te, m_te, n_rows)[test_slice]
    te_pr = average_precision_score(y_test, te)
    te_roc = roc_auc_score(y_test, te)
    rows.append((tag, max(vp), int(np.argmax(vp)) + 1, m.count_params(), te_pr, te_roc))
    print(f"{tag:<38} val PR {max(vp):.4f} @ep{int(np.argmax(vp))+1:<2} "
          f"params {m.count_params():>7,}  test PR {te_pr:.4f} ROC {te_roc:.4f}  "
          f"{time.time()-t0:>5.0f}s")

print("\nranked by validation PR-AUC (the selection metric):")
for tag, v, ep, p, te_pr, te_roc in sorted(rows, key=lambda r: -r[1]):
    print(f"  val {v:.4f}  test PR {te_pr:.4f}  {tag}")

# Write the results file the deck and the report read from. This script used to
# only print, so mtl_ablation.csv had to be reconstructed by hand from the log
# and went stale the moment the experiment was re-run.
import pandas as pd  # noqa: E402

out = config.METRICS_DIR / "mtl_ablation.csv"
pd.DataFrame([{
    "config": tag.split()[0],
    "description": tag.split(maxsplit=1)[1].strip(),
    "attention": "no attn" not in tag,
    "multi_task": "pit only" not in tag,
    "params": params,
    "best_epoch": ep,
    "val_pr_auc": round(float(v), 4),
    "test_pr_auc": round(float(te_pr), 4),
    "test_roc_auc": round(float(te_roc), 4),
} for tag, v, ep, params, te_pr, te_roc in rows]).to_csv(out, index=False)
print(f"\nsaved {out}")

d = {r[0][0]: r for r in rows}
print("\nablation readout (test PR-AUC):")
print(f"  attention effect, multi-task : A {d['A'][4]:.4f}  vs  C {d['C'][4]:.4f}"
      f"   -> {d['A'][4] - d['C'][4]:+.4f}")
print(f"  attention effect, pit only   : B {d['B'][4]:.4f}  vs  D {d['D'][4]:.4f}"
      f"   -> {d['B'][4] - d['D'][4]:+.4f}")
print(f"  multi-task effect, with attn : A {d['A'][4]:.4f}  vs  B {d['B'][4]:.4f}"
      f"   -> {d['A'][4] - d['B'][4]:+.4f}")
print(f"  multi-task effect, no attn   : C {d['C'][4]:.4f}  vs  D {d['D'][4]:.4f}"
      f"   -> {d['C'][4] - d['D'][4]:+.4f}")
