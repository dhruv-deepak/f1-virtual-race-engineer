"""
Phase 5 -- training and evaluation driver for Model 3.

Model 3 consumes grids rather than single windows, so it needs its own loader
and its own training loop. Everything *downstream* is deliberately shared with
Models 1 and 2: the same threshold-tuning procedure, the same metric set, the
same figures, and the same row in reports/metrics/comparison.csv.

The alignment that makes the comparison valid
---------------------------------------------
Model 3 predicts a grid of drivers at once. To be scored against Models 1 and 2
its pit head is scattered back to per-driver-lap rows, in exactly the order
sequences.npz holds them. The script asserts that alignment rather than trusting
it: after scattering, the labels it recovers for the test slice must equal
``y_test`` from sequences.npz element for element. If they do not, the run stops.

Usage
-----
    python -m src.models.train_mtl
    python -m src.models.train_mtl --epochs 5        # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import keras
import numpy as np

from src import config
from src.evaluation import metrics as M
from src.evaluation import plots
from src.models.mtl_architecture import DEFAULT_TASK_WEIGHTS, build_mtl_attention

warnings.filterwarnings("ignore")

MODEL_NAME = "mtl_attn"
DISPLAY = "Model 3 — Multi-task + cross-driver attention"


def load_grids():
    """Load the per-row tensors and the grid index that regroups them."""
    seq = np.load(config.DATA_PROCESSED / "sequences.npz")
    mtl_path = config.DATA_PROCESSED / "mtl_dataset.npz"
    if not mtl_path.exists():
        raise FileNotFoundError(
            f"{mtl_path} not found. Run `python -m src.features.build_mtl_dataset` first."
        )
    mtl = np.load(mtl_path, allow_pickle=False)
    with open(config.DATA_PROCESSED / "mtl_spec.json") as fh:
        spec = json.load(fh)

    X = np.concatenate([seq["X_train"], seq["X_val"], seq["X_test"]])
    n_train, n_val = len(seq["X_train"]), len(seq["X_val"])
    return seq, mtl, spec, X, n_train, n_val


def grid_arrays(mtl, X, split: str, spec):
    """
    Materialise one split as dense grids.

    Padded slots index row 0 and are then zeroed, so no real driver's window
    can bleed into a padding slot through the gather.
    """
    sel = mtl["grid_split"] == split
    grid = mtl["grid"][sel]
    mask = mtl["grid_mask"][sel].astype("float32")

    safe = np.where(grid >= 0, grid, 0)
    windows = X[safe] * mask[:, :, None, None]

    def gather(name, dtype):
        return (mtl[name][safe] * mask).astype(dtype) if dtype == "float32" \
            else np.where(mask > 0, mtl[name][safe], 0).astype(dtype)

    y = {
        "pit": gather("y_pit", "float32"),
        "compound": gather("y_compound", "int32"),
        "compound_mask": gather("y_compound_mask", "float32"),
        "laptime": gather("y_laptime", "float32"),
        "laptime_mask": gather("y_laptime_mask", "float32"),
        "outcome": gather("y_outcome", "int32"),
        "outcome_mask": gather("y_outcome_mask", "float32"),
        "driver_mask": mask,
    }
    x = {"windows": windows.astype("float32"), "driver_mask": mask}
    return x, y, grid, mask


def scatter_to_rows(pred_grid, grid, mask, n_rows):
    """Flatten grid predictions back to one value per driver-lap row."""
    out = np.full(n_rows, np.nan, dtype="float32")
    valid = mask > 0
    out[grid[valid]] = pred_grid[valid]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Model 3.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    config.ensure_dirs()
    config.set_seed()
    plots.apply_style()

    seq, mtl, spec, X, n_train, n_val = load_grids()
    n_rows = len(X)

    x_tr, y_tr, _, _ = grid_arrays(mtl, X, "train", spec)
    x_va, y_va, g_va, m_va = grid_arrays(mtl, X, "val", spec)
    x_te, y_te, g_te, m_te = grid_arrays(mtl, X, "test", spec)

    print(f"\n{DISPLAY}")
    print("=" * 70)
    for name, xx, mm in [("train", x_tr, x_tr["driver_mask"]),
                         ("val", x_va, m_va), ("test", x_te, m_te)]:
        print(f"{name:<5} grids {xx['windows'].shape[0]:>6,}  "
              f"driver-laps {int(mm.sum()):>7,}  "
              f"drivers/grid {mm.sum(1).mean():.1f}")
    print(f"window {spec['seq_len']} laps | features {spec['n_features']} | "
          f"max drivers {spec['max_drivers']}")

    pit_tr = y_tr["pit"][x_tr["driver_mask"] > 0]
    pw = float((len(pit_tr) - pit_tr.sum()) / pit_tr.sum())
    print(f"positive class weight: {pw:.2f}")
    print(f"task weights: {DEFAULT_TASK_WEIGHTS}")

    model = build_mtl_attention(
        seq_len=spec["seq_len"], n_features=spec["n_features"],
        max_drivers=spec["max_drivers"], n_compounds=spec["n_compound_classes"],
        n_outcomes=spec["n_outcome_classes"], pos_weight=pw, learning_rate=args.lr,
    )
    model.summary()

    t0 = time.perf_counter()
    history = model.fit(
        x_tr, y_tr, validation_data=(x_va, y_va),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                          patience=config.EARLY_STOPPING_PATIENCE,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max",
                                              factor=0.5, patience=4, min_lr=1e-5,
                                              verbose=1),
        ],
    )
    train_time = time.perf_counter() - t0
    epochs_run = len(history.history["total_loss"])
    print(f"\ntrained {epochs_run} epochs in {train_time:.1f}s")

    # ---------------- back to per-row predictions ----------------
    val_grid = model.predict(x_va, batch_size=64, verbose=0)["pit"][..., 0]
    test_grid = model.predict(x_te, batch_size=64, verbose=0)["pit"][..., 0]
    val_rows = scatter_to_rows(val_grid, g_va, m_va, n_rows)
    test_rows = scatter_to_rows(test_grid, g_te, m_te, n_rows)

    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n_rows)
    val_prob = val_rows[val_slice]
    test_prob = test_rows[test_slice]

    # The alignment assertion: recovered labels must match sequences.npz exactly.
    if np.isnan(val_prob).any() or np.isnan(test_prob).any():
        raise ValueError("some rows received no prediction -- grid coverage is incomplete")
    recovered = mtl["y_pit"][test_slice]
    if not np.array_equal(recovered.astype(np.int8), seq["y_test"].astype(np.int8)):
        raise ValueError("row alignment is wrong -- recovered labels differ from y_test")
    print("alignment verified: scattered rows match the Model 1/2 test set exactly")

    y_val, y_test = seq["y_val"], seq["y_test"]

    # ---------------- identical downstream evaluation ----------------
    threshold, val_f1 = M.tune_threshold(y_val, val_prob)
    print(f"tuned threshold on validation: {threshold:.3f} (val F1 {val_f1:.4f})")

    test_default = M.compute(y_test, test_prob, 0.5)
    test_metrics = M.compute(y_test, test_prob, threshold)
    baseline = M.majority_baseline(y_test)
    print(M.format_report("Never-pit baseline (test)", baseline))
    print(M.format_report(f"{DISPLAY} @ threshold 0.50 (test)", test_default))
    print(M.format_report(f"{DISPLAY} @ tuned threshold (test)", test_metrics))

    # plots.plot_* expect 'loss'/'accuracy'; the multi-task history names the
    # combined objective 'total_loss'.
    hist = dict(history.history)
    hist["loss"], hist["val_loss"] = hist["total_loss"], hist["val_total_loss"]

    c = M.curves(y_test, test_prob)
    written = [
        plots.plot_accuracy(hist, MODEL_NAME, DISPLAY),
        plots.plot_loss(hist, MODEL_NAME, DISPLAY),
        plots.plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), MODEL_NAME, DISPLAY),
        plots.plot_roc(c["fpr"], c["tpr"], test_metrics["roc_auc"], MODEL_NAME, DISPLAY),
        plots.plot_precision_recall(c["recall"], c["precision"], test_metrics["pr_auc"],
                                    float(y_test.mean()), MODEL_NAME, DISPLAY),
    ]

    # Latency is measured per driver-lap, not per grid, so it is comparable with
    # the single-driver models.
    sample = {k: v[:256] for k, v in x_te.items()}
    model.predict({k: v[:8] for k, v in x_te.items()}, verbose=0)
    t1 = time.perf_counter()
    model.predict(sample, batch_size=64, verbose=0)
    per_lap_ms = (time.perf_counter() - t1) * 1000.0 / max(sample["driver_mask"].sum(), 1)

    payload = {
        "model": MODEL_NAME, "display_name": DISPLAY,
        "n_parameters": int(model.count_params()),
        "train_time_s": round(train_time, 2), "epochs_run": epochs_run,
        "inference_ms_per_sample": round(float(per_lap_ms), 4),
        "seq_len": spec["seq_len"], "horizon": spec["horizon"],
        "n_features": spec["n_features"], "max_drivers": spec["max_drivers"],
        "pos_weight": round(pw, 4), "task_weights": DEFAULT_TASK_WEIGHTS,
        "tuned_threshold": threshold,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "validation": M.compute(y_val, val_prob, threshold),
        "test": test_metrics,
        "test_at_threshold_0.5": test_default,
        "baseline_never_pit": baseline,
    }
    json_path = M.save(MODEL_NAME, payload)
    model.save_weights(str(config.MODELS_DIR / f"{MODEL_NAME}.weights.h5"))

    print("\nArtefacts written")
    print(f"  metrics  {json_path}")
    for w in written:
        print(f"  figure   {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
