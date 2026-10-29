"""
Phase 6 -- three-model comparison artefacts.

Produces the figures and the table that Review 3 asks for: the three models'
ROC and precision-recall curves overlaid, their confusion matrices side by side,
their training curves, an accuracy-versus-cost view, and one metrics table.

A note on the loss comparison
-----------------------------
Models 1 and 2 minimise weighted binary cross-entropy. Model 3 minimises a
*combined* objective across four heads, so its total loss is on a different
scale and overlaying it would be meaningless. The comparison therefore plots
Model 3's ``pit_loss`` component, which is the same weighted cross-entropy on
the same task, and says so on the figure.

Usage
-----
    python -m src.evaluation.compare
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config
from src.evaluation import plots
from src.evaluation.plots import (AXIS, INK_MUTED, INK_PRIMARY,
                                  INK_SECONDARY, SEQUENTIAL, SURFACE)

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-3 of the project palette: validated colourblind-safe as a
# set, and fixed so a model keeps the same colour across every figure.
SERIES = {
    "bilstm": "#2a78d6",
    "cnn_bilstm": "#eb6834",
    "mtl_attn": "#1baf7a",
}
SHORT = {
    "bilstm": "Model 1 — Bi-LSTM",
    "cnn_bilstm": "Model 2 — CNN–BiLSTM",
    "mtl_attn": "Model 3 — MTL + attention",
}
ORDER = ["bilstm", "cnn_bilstm", "mtl_attn"]
DPI = 200


def load_all():
    """Metrics, histories and recovered probabilities for every model."""
    seq = np.load(config.DATA_PROCESSED / "sequences.npz")
    models = {}
    for name in ORDER:
        with open(config.METRICS_DIR / f"{name}_metrics.json") as fh:
            m = json.load(fh)
        probs = np.load(config.METRICS_DIR / f"{name}_probs.npz")
        models[name] = {"metrics": m, "test": probs["test"], "val": probs["val"]}
    return seq["y_test"], models


def _save(fig, stem):
    path = config.FIGURES_DIR / f"comparison_{stem}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Threshold-independent curves, overlaid
# ---------------------------------------------------------------------------
def plot_roc(y_test, models):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", zorder=0)
    ax.annotate("chance", xy=(0.72, 0.68), fontsize=8, color=INK_MUTED, rotation=32)
    for name in ORDER:
        fpr, tpr, _ = roc_curve(y_test, models[name]["test"])
        auc = models[name]["metrics"]["test"]["roc_auc"]
        ax.plot(fpr, tpr, color=SERIES[name], label=f"{SHORT[name]}  ({auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — all three models, 2024 test set")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    return _save(fig, "roc")


def plot_pr(y_test, models):
    base = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.axhline(base, color=AXIS, lw=1, ls="--", zorder=0)
    ax.annotate(f"random baseline = {base:.3f}", xy=(0.03, base + 0.015),
                fontsize=8, color=INK_MUTED)
    for name in ORDER:
        prec, rec, _ = precision_recall_curve(y_test, models[name]["test"])
        ap = models[name]["metrics"]["test"]["pr_auc"]
        ax.plot(rec, prec, color=SERIES[name], label=f"{SHORT[name]}  ({ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves — the honest view under 8.5% positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    return _save(fig, "precision_recall")


# ---------------------------------------------------------------------------
# Confusion matrices, side by side
# ---------------------------------------------------------------------------
def plot_confusions(models):
    labels = ("No pit", "Pit ≤3 laps")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for ax, name in zip(axes, ORDER):
        cm = np.array(models[name]["metrics"]["test"]["confusion_matrix"])
        pct = cm / np.clip(cm.sum(1, keepdims=True), 1, None) * 100
        im = ax.imshow(pct, cmap=SEQUENTIAL, vmin=0, vmax=100)
        ax.set_xticks([0, 1], labels=labels)
        ax.set_yticks([0, 1], labels=labels if name == ORDER[0] else ["", ""])
        ax.set_xlabel("Predicted")
        if name == ORDER[0]:
            ax.set_ylabel("Actual")
        thr = models[name]["metrics"]["tuned_threshold"]
        ax.set_title(f"{SHORT[name]}\nthreshold {thr:.3f}", fontsize=10.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(2):
            for j in range(2):
                colour = SURFACE if pct[i, j] > 55 else INK_PRIMARY
                ax.text(j, i - 0.06, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=colour)
                ax.text(j, i + 0.20, f"{pct[i, j]:.1f}%", ha="center", va="center",
                        fontsize=8.5, color=colour)
    cbar = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.015)
    cbar.set_label("% of actual class", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    return _save(fig, "confusion_matrices")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def _history(models, name, key):
    h = models[name]["metrics"]["history"]
    # Model 3's combined objective is not on the same scale as the other two;
    # its pit head is, and that is the like-for-like quantity.
    if key == "loss" and "pit_loss" in h:
        return h["pit_loss"], h["val_pit_loss"]
    return h[key], h[f"val_{key}"]


def plot_curves(models, key, title, ylabel, stem, note=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for name in ORDER:
        _, val = _history(models, name, key)
        ax.plot(np.arange(1, len(val) + 1), val, color=SERIES[name], label=SHORT[name])
        best = int(np.argmin(val)) if key == "loss" else int(np.argmax(val))
        ax.plot([best + 1], [val[best]], "o", ms=6, color=SERIES[name],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    if note:
        ax.annotate(note, xy=(0, -0.22), xycoords="axes fraction",
                    fontsize=8, color=INK_MUTED, va="top")
    return _save(fig, stem)


# ---------------------------------------------------------------------------
# Accuracy against cost
# ---------------------------------------------------------------------------
def plot_efficiency(models):
    """
    PR-AUC against parameter count.

    'Most efficient' is a two-axis claim: a model that gives up little accuracy
    for a large saving in size or latency may well be the right one to deploy in
    a system whose premise is real-time inference.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    params = [models[n]["metrics"]["n_parameters"] for n in ORDER]
    scores = [models[n]["metrics"]["test"]["pr_auc"] for n in ORDER]
    base = models[ORDER[0]]["metrics"]["baseline_never_pit"]["pr_auc"]

    # Generous padding first, so label placement can be decided against the
    # final axes rather than against autoscaled limits.
    ax.set_xscale("log")
    ax.set_xlim(min(params) / 2.6, max(params) * 2.6)
    top = max(scores) + 0.045
    ax.set_ylim(min(base - 0.02, 0.24), top)

    for name in ORDER:
        m = models[name]["metrics"]
        x, y = m["n_parameters"], m["test"]["pr_auc"]
        ax.scatter(x, y, s=130, color=SERIES[name], zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        # Labels go below any point sitting near the top, so none of them
        # collide with the title.
        above = y < top - 0.04
        ax.annotate(f"{SHORT[name]}\n{m['inference_ms_per_sample']:.3f} ms/lap",
                    xy=(x, y), xytext=(0, 14 if above else -30),
                    textcoords="offset points", ha="center",
                    va="bottom" if above else "top",
                    fontsize=8.5, color=INK_SECONDARY)

    ax.axhline(base, color=AXIS, lw=1, ls="--", zorder=0)
    ax.annotate(f"random baseline = {base:.3f}", xy=(0.02, base + 0.004),
                xycoords=("axes fraction", "data"), fontsize=8, color=INK_MUTED)
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Accuracy against model size")
    ax.set_axisbelow(True)
    return _save(fig, "efficiency")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def build_table(y_test, models) -> pd.DataFrame:
    base = models[ORDER[0]]["metrics"]["baseline_never_pit"]
    rows = [{
        "model": "Never-pit baseline", "params": None, "train_s": None,
        "ms_per_lap": None, "threshold": 0.50,
        **{k: base[k] for k in ("accuracy", "balanced_accuracy", "precision",
                                "recall", "f1", "roc_auc", "pr_auc")},
    }]
    for name in ORDER:
        m = models[name]["metrics"]
        rows.append({
            "model": SHORT[name], "params": m["n_parameters"],
            "train_s": m["train_time_s"], "ms_per_lap": m["inference_ms_per_sample"],
            "threshold": round(m["tuned_threshold"], 3),
            **{k: m["test"][k] for k in ("accuracy", "balanced_accuracy", "precision",
                                         "recall", "f1", "roc_auc", "pr_auc")},
        })
    return pd.DataFrame(rows).round(4)


def main() -> int:
    config.ensure_dirs()
    plots.apply_style()

    y_test, models = load_all()
    print(f"test set: {len(y_test):,} driver-laps, {100 * y_test.mean():.2f}% positive\n")

    written = [
        plot_roc(y_test, models),
        plot_pr(y_test, models),
        plot_confusions(models),
        plot_curves(models, "accuracy", "Validation accuracy per epoch",
                    "Validation accuracy", "accuracy"),
        plot_curves(models, "loss", "Validation loss per epoch",
                    "Weighted cross-entropy", "loss",
                    note="Model 3 shows its pit-head loss: its combined multi-task "
                         "objective is on a different scale and would not be comparable."),
        plot_efficiency(models),
    ]

    table = build_table(y_test, models)
    csv_path = config.METRICS_DIR / "three_model_comparison.csv"
    table.to_csv(csv_path, index=False)
    md_path = config.METRICS_DIR / "three_model_comparison.md"
    md_path.write_text(table.to_markdown(index=False), encoding="utf-8")

    print(table.to_string(index=False))
    best_pr = table.iloc[1:].sort_values("pr_auc", ascending=False).iloc[0]
    smallest = table.iloc[1:].sort_values("params").iloc[0]
    print(f"\nbest PR-AUC : {best_pr['model']} ({best_pr['pr_auc']:.4f})")
    print(f"smallest    : {smallest['model']} ({int(smallest['params']):,} params, "
          f"{smallest['pr_auc']:.4f} PR-AUC)")

    print("\nArtefacts written")
    for w in written:
        print(f"  figure  {w}")
    print(f"  table   {csv_path}")
    print(f"  table   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
