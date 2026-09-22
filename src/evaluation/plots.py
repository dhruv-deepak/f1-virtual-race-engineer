"""
Figure generation for model evaluation.

Produces the five figures each model reports: training/validation accuracy,
training/validation loss, confusion matrix, ROC curve and precision-recall
curve. Every figure is saved to ``reports/figures/`` as a 200 dpi PNG named
``<model>_<figure>.png`` so the three models can be compared side by side.

Design notes
------------
* Series colours come from a fixed categorical order (blue = train,
  orange = validation) and are never cycled; the same colour always means the
  same thing across all three models' figures.
* The confusion matrix uses a **single-hue** light-to-dark blue ramp, because
  cell counts are a continuous magnitude. A rainbow map would imply categories
  that do not exist and is not colourblind-safe.
* Grid and axes are deliberately recessive so the data is the darkest thing on
  the page; labels and values wear ink colours, never the series colour.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")  # figures are written to disk, never displayed interactively
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402

# --- design tokens ---------------------------------------------------------
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_TRAIN = "#2a78d6"   # categorical slot 1 (blue)
SERIES_VAL = "#eb6834"     # categorical slot 2 (orange)

# Single-hue sequential ramp, steps 100 -> 700 of the blue scale.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("f1_blue", BLUE_RAMP)

DPI = 200


def apply_style() -> None:
    """Global matplotlib defaults matching the project's figure style."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "font.size": 10,
        "figure.dpi": 110,
    })


def _save(fig, model_name: str, figure_name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{model_name}_{figure_name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_curve(history: dict, metric: str, model_name: str,
               title: str, ylabel: str, figure_name: str) -> str:
    """One training curve: the metric on train vs. validation, per epoch."""
    train = history[metric]
    val = history.get(f"val_{metric}", [])
    epochs = np.arange(1, len(train) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, train, color=SERIES_TRAIN, label="Training")
    if len(val):
        ax.plot(epochs, val, color=SERIES_VAL, label="Validation")
        # Mark the epoch the early-stopping callback restored weights from.
        best = int(np.argmin(val)) if metric == "loss" else int(np.argmax(val))
        ax.axvline(best + 1, color=AXIS, linewidth=1, linestyle="--", zorder=0)
        ax.annotate(f"best epoch {best + 1}", xy=(best + 1, val[best]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=INK_MUTED)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_axisbelow(True)
    # Epochs are whole numbers; the default locator happily prints "1.25".
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    return _save(fig, model_name, figure_name)


def plot_accuracy(history: dict, model_name: str, display_name: str) -> str:
    return plot_curve(history, "accuracy", model_name,
                      f"{display_name} — Accuracy per epoch", "Accuracy", "accuracy")


def plot_loss(history: dict, model_name: str, display_name: str) -> str:
    return plot_curve(history, "loss", model_name,
                      f"{display_name} — Loss per epoch", "Weighted cross-entropy", "loss")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, model_name: str, display_name: str,
                          labels: tuple[str, str] = ("No pit", "Pit within 3 laps"),
                          figure_name: str = "confusion_matrix") -> str:
    """
    Confusion matrix with raw counts and row-normalised percentages.

    Each cell shows the count and, beneath it, that count as a share of its true
    class. Row percentages are what matter here: with an imbalanced test set the
    raw counts in the 'No pit' row dwarf everything else, and only the
    normalised view shows whether the model actually finds pit stops.
    """
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, np.where(row_sums == 0, 1, row_sums)) * 100

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(pct, cmap=SEQUENTIAL, vmin=0, vmax=100)

    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{display_name} — Confusion matrix (test set)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Light text on the dark end of the ramp, dark ink on the light end.
            colour = SURFACE if pct[i, j] > 55 else INK_PRIMARY
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=colour)
            ax.text(j, i + 0.22, f"{pct[i, j]:.1f}% of actual", ha="center", va="center",
                    fontsize=8.5, color=colour)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of actual class", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    return _save(fig, model_name, figure_name)


# ---------------------------------------------------------------------------
# Threshold-independent curves
# ---------------------------------------------------------------------------
def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float,
             model_name: str, display_name: str) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    # Chance line is chrome, not a series -- it wears the axis colour.
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", zorder=0)
    ax.plot(fpr, tpr, color=SERIES_TRAIN)
    ax.annotate(f"ROC-AUC = {auc:.3f}", xy=(0.55, 0.12),
                fontsize=11, color=INK_PRIMARY, fontweight="bold")
    ax.annotate("chance", xy=(0.72, 0.68), fontsize=8, color=INK_MUTED, rotation=32)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{display_name} — ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_axisbelow(True)
    return _save(fig, model_name, "roc")


def plot_precision_recall(recall: np.ndarray, precision: np.ndarray, ap: float,
                          positive_rate: float, model_name: str, display_name: str) -> str:
    """
    Precision-recall curve.

    The reference line is the positive class rate, which is what a model that
    guesses at random achieves. On an imbalanced problem this is the honest
    baseline -- far more informative than ROC's diagonal.
    """
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.axhline(positive_rate, color=AXIS, linewidth=1, linestyle="--", zorder=0)
    ax.plot(recall, precision, color=SERIES_TRAIN)
    ax.annotate(f"PR-AUC = {ap:.3f}", xy=(0.05, 0.12),
                fontsize=11, color=INK_PRIMARY, fontweight="bold")
    ax.annotate(f"random baseline = {positive_rate:.3f}",
                xy=(0.05, positive_rate + 0.02), fontsize=8, color=INK_MUTED)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{display_name} — Precision-recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_axisbelow(True)
    return _save(fig, model_name, "precision_recall")
