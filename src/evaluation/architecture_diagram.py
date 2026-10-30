"""
Phase 7 -- system architecture diagram for the final review deck.

Draws the whole pipeline in one figure: acquisition, feature engineering,
windowing, the three models, and evaluation. Rendered rather than drawn by hand
so it stays in step with the code -- the parameter counts are read from the
committed metrics files.

Layout note
-----------
All geometry is in axes fractions, and the vertical constants are derived from
the figure height rather than guessed: ``_line(pt)`` converts a font size into
the fraction of the figure that one line of that size occupies, so text cannot
silently outgrow its box when the figure is resized.

Usage
-----
    python -m src.evaluation.architecture_diagram
"""

from __future__ import annotations

import json
import sys
import warnings

from src import config
from src.evaluation import plots
from src.evaluation.plots import AXIS, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Panel fills, matching the slide decks so the diagram sits naturally among them.
CARD = "#f8fafc"
CHIP = "#e1e1ea"
CHIP_MID = "#c7c7d0"
SERIES = {"bilstm": "#2a78d6", "cnn_bilstm": "#eb6834", "mtl_attn": "#1baf7a"}

FIG_W, FIG_H = 13.2, 7.0
TITLE_PT, BODY_PT = 10.5, 8.0
DPI = 200


def _line(pt, lead=1.45):
    """Fraction of figure height occupied by one line of `pt` type."""
    return pt / 72.0 * lead / FIG_H


LINE = _line(BODY_PT)
TITLE_DROP = _line(TITLE_PT) * 1.15


def box(ax, x, y, w, h, title, lines, fill=CARD, edge=CHIP_MID, accent=None):
    """Rounded panel with a bold title and body lines laid out from the top."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018",
                                facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=2))
    top = y + h
    ax.text(x + w / 2, top - TITLE_DROP, title, ha="center", va="center",
            fontsize=TITLE_PT, fontweight="bold", color=accent or INK_PRIMARY, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - TITLE_DROP - LINE * (1.35 + i), line, ha="center",
                va="center", fontsize=BODY_PT, color=INK_SECONDARY, zorder=3)


def section(ax, y, text):
    """Stage label. The opaque background masks the connector passing behind it."""
    ax.text(0.0, y, text, fontsize=8.5, fontweight="bold", color=INK_MUTED, zorder=4,
            va="center", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.5))


def arrow(ax, x1, y1, x2, y2, color=AXIS, lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color,
                                 linewidth=lw, mutation_scale=12,
                                 shrinkA=1, shrinkB=1, zorder=1))


def main() -> int:
    config.ensure_dirs()
    plots.apply_style()

    params = {}
    for name in SERIES:
        with open(config.METRICS_DIR / f"{name}_metrics.json") as fh:
            params[name] = json.load(fh)["n_parameters"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    # ---- row 1: the data pipeline, left to right -------------------------
    section(ax, 0.955, "1   DATA PIPELINE")
    stages = [
        ("FastF1 acquisition", ["Official F1 timing feed", "2021–2024 · 90 races",
                                "3.7 Hz telemetry reduced", "to 8 channels per lap"]),
        ("Feature engineering", ["55 features per driver-lap", "timing · tyre · position",
                                 "track status · weather", "telemetry · competitor"]),
        ("Leak-free labelling", ["Pits within next 3 laps", "forward-shifted, so lap t's",
                                 "own stop cannot enter", "its own label"]),
        ("Sequence dataset", ["10-lap windows per driver", "81,270 samples",
                              "train 21–22 · val 23 · test 24", "scaler fit on train only"]),
    ]
    w1, gap1 = 0.232, 0.256
    for i, (t, lines) in enumerate(stages):
        box(ax, i * gap1, 0.715, w1, 0.215, t, lines, fill=CHIP if i == 0 else CARD)
        if i < len(stages) - 1:
            arrow(ax, i * gap1 + w1, 0.8225, (i + 1) * gap1, 0.8225)

    # ---- row 2: the three models ----------------------------------------
    section(ax, 0.664, "2   MODELS — one task, three architectures")
    models = [
        ("bilstm", "Model 1 · Bi-LSTM",
         ["Bidirectional LSTM 64 → 32", "dropout 0.4 · light L2", "Dense → sigmoid"]),
        ("cnn_bilstm", "Model 2 · CNN–BiLSTM",
         ["Conv1D ×2 + BatchNorm", "→ bidirectional LSTM 16", "no pooling: only 10 laps"]),
        ("mtl_attn", "Model 3 · MTL + attention",
         ["Shared per-driver encoder", "→ attention across drivers", "→ four task heads"]),
    ]
    w2, gap2 = 0.312, 0.344
    for i, (key, t, lines) in enumerate(models):
        box(ax, i * gap2, 0.432, w2, 0.196, t, lines + [f"{params[key]:,} parameters"],
            edge=SERIES[key], accent=SERIES[key])
        arrow(ax, i * gap2 + w2 / 2, 0.715, i * gap2 + w2 / 2, 0.628, color=CHIP_MID)

    # ---- row 3: evaluation ----------------------------------------------
    section(ax, 0.356, "3   EVALUATION — identical for all three")
    evals = [
        ("Shared training protocol", ["Weighted cross-entropy", "early stop on val PR-AUC",
                                      "threshold tuned on validation"]),
        ("Metrics on 2024", ["Accuracy · precision · recall", "F1 · ROC-AUC · PR-AUC",
                             "confusion matrix per model"]),
        ("Historical race replay", ["644 real stops replayed", "lead time per anticipated stop",
                                    "false alarms per driver-race"]),
    ]
    for i, (t, lines) in enumerate(evals):
        box(ax, i * gap2, 0.135, w2, 0.185, t, lines)
        arrow(ax, i * gap2 + w2 / 2, 0.432, i * gap2 + w2 / 2, 0.320, color=CHIP_MID)

    # ---- footer ----------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.098,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                facecolor=CHIP, edgecolor="none", zorder=2))
    ax.text(0.018, 0.071, "What makes the three-way comparison valid", fontsize=9,
            fontweight="bold", color=INK_PRIMARY, va="center", zorder=3)
    ax.text(0.018, 0.032,
            "All three models consume identical feature tensors, the identical train-only scaler "
            "and identical season splits. Model 3 regroups those same rows into (race, lap)\ngrids "
            "so attention can run across drivers, then scatters its predictions back to "
            "per-driver-lap rows — an alignment the training script asserts rather than assumes.",
            fontsize=7.6, color=INK_SECONDARY, va="center", zorder=3, linespacing=1.7)

    ax.set_title("Deep Learning-Based Virtual Race Engineer — System Architecture",
                 fontsize=13.5, fontweight="bold", color=INK_PRIMARY, pad=12)

    out = config.FIGURES_DIR / "architecture_diagram.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
