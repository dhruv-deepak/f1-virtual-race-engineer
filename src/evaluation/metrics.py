"""
Metric computation and reporting.

Keeps every number the project reports in one place, so all three models are
scored identically. Results are written to ``reports/metrics/`` as per-model
JSON plus an appended row in ``comparison.csv``, which becomes the Review 3
comparison table.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

COMPARISON_CSV = "comparison.csv"


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Pick the decision threshold that maximises F1 **on the validation set**.

    0.5 is only the right cut-off when the classes are balanced. Tuning on
    validation and then applying that fixed threshold to the test set keeps the
    test set untouched, which a threshold tuned on test would not.

    Guard: candidates are restricted to thresholds whose precision beats the
    base rate. Without it, a model with no real signal maximises F1 by calling
    a pit stop on every single lap -- recall goes to 1.0, precision falls to the
    base rate, and the "best" threshold is a degenerate one. If nothing clears
    the base rate the model has learned nothing useful and we keep 0.5, so the
    failure stays visible in the confusion matrix instead of being hidden.
    """
    base_rate = float(np.mean(y_true))
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) == 0:
        return 0.5, 0.0

    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    usable = precision > base_rate
    if not usable.any():
        return 0.5, 0.0

    f1 = np.where(usable, f1, -np.inf)
    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])


def compute(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Every metric the project reports, for one model on one split."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Mean of per-class recall: unlike accuracy, this cannot be gamed by
        # always predicting the majority class.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def majority_baseline(y_true: np.ndarray) -> dict:
    """
    The "never pit" baseline.

    Included in every report because it is the number that exposes how little
    raw accuracy means here: predicting no pit stop ever scores roughly 87%
    accuracy while never once calling a stop.
    """
    y_true = np.asarray(y_true).ravel()
    return compute(y_true, np.zeros_like(y_true, dtype=float), threshold=0.5)


def curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Points for the ROC and precision-recall figures."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}


def save(model_name: str, payload: dict) -> str:
    """Write per-model JSON and append a row to the shared comparison table."""
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.METRICS_DIR / f"{model_name}_metrics.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)

    test = payload["test"]
    row = {
        "model": payload.get("display_name", model_name),
        "params": payload.get("n_parameters"),
        "train_time_s": payload.get("train_time_s"),
        "epochs_run": payload.get("epochs_run"),
        "inference_ms_per_sample": payload.get("inference_ms_per_sample"),
        "accuracy": test["accuracy"],
        "balanced_accuracy": test["balanced_accuracy"],
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "roc_auc": test["roc_auc"],
        "pr_auc": test["pr_auc"],
    }

    csv_path = config.METRICS_DIR / COMPARISON_CSV
    df = pd.DataFrame([row])
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        prev = prev[prev["model"] != row["model"]]      # overwrite a re-run
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return str(path)


def format_report(name: str, m: dict) -> str:
    """Human-readable block for the console and the notebook."""
    cm = np.array(m["confusion_matrix"])
    return (
        f"\n{name}\n{'-' * len(name)}\n"
        f"  samples            {m['n_samples']:,}  (positives {100 * m['positive_rate']:.2f}%)\n"
        f"  threshold          {m['threshold']:.3f}\n"
        f"  accuracy           {m['accuracy']:.4f}\n"
        f"  balanced accuracy  {m['balanced_accuracy']:.4f}\n"
        f"  precision          {m['precision']:.4f}\n"
        f"  recall             {m['recall']:.4f}\n"
        f"  f1                 {m['f1']:.4f}\n"
        f"  roc-auc            {m['roc_auc']:.4f}\n"
        f"  pr-auc             {m['pr_auc']:.4f}\n"
        f"  confusion matrix   [[{cm[0, 0]:>6,} {cm[0, 1]:>6,}]   <- actual no-pit\n"
        f"                      [{cm[1, 0]:>6,} {cm[1, 1]:>6,}]]  <- actual pit\n"
    )
