"""
Phase 3 -- training and evaluation driver.

Trains one architecture on the Phase 2 dataset and produces everything the
review needs: metrics, the accuracy and loss curves, the confusion matrix, and
the ROC/PR curves.

The model is the only thing that varies. Data, splits, seed, loss, callbacks,
threshold-tuning procedure and metrics are held fixed across every run, which is
what makes the three-model comparison in Review 3 a fair one.

Usage
-----
    python -m src.models.train --model bilstm
    python -m src.models.train --model bilstm --epochs 5     # quick check
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
from src.models import architectures

warnings.filterwarnings("ignore")

DISPLAY_NAMES = {
    "bilstm": "Model 1 — Bi-LSTM",
    "cnn_bilstm": "Model 2 — CNN-BiLSTM",
    "mtl_attn": "Model 3 — Multi-task + cross-driver attention",
}


def load_dataset():
    """Load the tensors written by ``src.features.build_dataset``."""
    npz_path = config.DATA_PROCESSED / "sequences.npz"
    meta_path = config.DATA_PROCESSED / "feature_names.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run `python -m src.features.build_dataset` first."
        )
    data = np.load(npz_path)
    with open(meta_path) as fh:
        meta = json.load(fh)
    return data, meta


def positive_weight(y_train: np.ndarray) -> float:
    """
    How much to up-weight a positive lap so the two classes contribute equally.

    With a 13% positive rate this comes out near 6.7, i.e. one upcoming pit stop
    counts as much as roughly seven quiet laps.
    """
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    return neg / max(pos, 1.0)


def make_callbacks(model_name: str):
    """
    Early stopping on validation PR-AUC, not validation loss.

    Under class imbalance the loss can keep creeping down while the model
    quietly gets worse at the thing we care about -- actually identifying pit
    stops. PR-AUC tracks that directly.
    """
    ckpt = config.MODELS_DIR / f"{model_name}.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc", mode="max", factor=0.5, patience=4,
            min_lr=1e-5, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_pr_auc", mode="max",
            save_best_only=True, verbose=0,
        ),
    ]


def measure_inference(model, X: np.ndarray, n: int = 2048) -> float:
    """
    Milliseconds to score one lap.

    The system claims real-time operation, so latency is part of what 'most
    efficient model' means -- not just accuracy.
    """
    sample = X[:n]
    model.predict(sample[:32], verbose=0)          # warm up the graph
    t0 = time.perf_counter()
    model.predict(sample, batch_size=256, verbose=0)
    return (time.perf_counter() - t0) * 1000.0 / len(sample)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate one model.")
    parser.add_argument("--model", default="bilstm", choices=sorted(architectures.BUILDERS))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    config.ensure_dirs()
    config.set_seed()
    plots.apply_style()

    name = args.model
    display = DISPLAY_NAMES.get(name, name)

    # ---------------- data ----------------
    data, meta = load_dataset()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = meta["features"]

    print(f"\n{display}")
    print("=" * 70)
    print(f"train {X_train.shape}  positives {100 * y_train.mean():.2f}%")
    print(f"val   {X_val.shape}  positives {100 * y_val.mean():.2f}%")
    print(f"test  {X_test.shape}  positives {100 * y_test.mean():.2f}%")
    print(f"features: {len(feature_names)} | window: {meta['seq_len']} laps "
          f"| horizon: {meta['horizon']} laps")

    # ---------------- model ----------------
    pw = positive_weight(y_train)
    print(f"\npositive class weight: {pw:.2f}")
    model = architectures.build(
        name, input_shape=(X_train.shape[1], X_train.shape[2]),
        pos_weight=pw, learning_rate=args.lr,
    )
    model.summary()

    # ---------------- train ----------------
    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=make_callbacks(name),
        verbose=2,
    )
    train_time = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])
    print(f"\ntrained {epochs_run} epochs in {train_time:.1f}s")

    # ---------------- threshold ----------------
    val_prob = model.predict(X_val, batch_size=512, verbose=0).ravel()
    threshold, val_f1 = M.tune_threshold(y_val, val_prob)
    print(f"tuned threshold on validation: {threshold:.3f} (val F1 {val_f1:.4f})")

    # ---------------- evaluate ----------------
    test_prob = model.predict(X_test, batch_size=512, verbose=0).ravel()
    val_metrics = M.compute(y_val, val_prob, threshold)
    test_default = M.compute(y_test, test_prob, 0.5)
    test_metrics = M.compute(y_test, test_prob, threshold)
    baseline = M.majority_baseline(y_test)

    print(M.format_report("Never-pit baseline (test)", baseline))
    print(M.format_report(f"{display} @ threshold 0.50 (test)", test_default))
    print(M.format_report(f"{display} @ tuned threshold (test)", test_metrics))

    # ---------------- figures ----------------
    c = M.curves(y_test, test_prob)
    written = [
        plots.plot_accuracy(history.history, name, display),
        plots.plot_loss(history.history, name, display),
        plots.plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), name, display),
        plots.plot_roc(c["fpr"], c["tpr"], test_metrics["roc_auc"], name, display),
        plots.plot_precision_recall(c["recall"], c["precision"], test_metrics["pr_auc"],
                                    float(y_test.mean()), name, display),
    ]

    # ---------------- persist ----------------
    payload = {
        "model": name,
        "display_name": display,
        "n_parameters": int(model.count_params()),
        "train_time_s": round(train_time, 2),
        "epochs_run": epochs_run,
        "inference_ms_per_sample": round(measure_inference(model, X_test), 4),
        "seq_len": meta["seq_len"],
        "horizon": meta["horizon"],
        "n_features": len(feature_names),
        "pos_weight": round(pw, 4),
        "tuned_threshold": threshold,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "validation": val_metrics,
        "test": test_metrics,
        "test_at_threshold_0.5": test_default,
        "baseline_never_pit": baseline,
    }
    json_path = M.save(name, payload)

    print("\nArtefacts written")
    print(f"  metrics  {json_path}")
    for w in written:
        print(f"  figure   {w}")
    print(f"  weights  {config.MODELS_DIR / (name + '.keras')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
