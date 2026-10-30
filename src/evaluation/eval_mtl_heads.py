"""
Phase 7 -- evaluate Model 3's three auxiliary heads.

Model 3 predicts four things per driver per lap: whether the driver pits within
three laps, which compound they will fit at that stop, their next lap's pace,
and their final race outcome. Only the pit head had been scored, so the other
three were trained but unvalidated and could not be claimed.

This module scores them on the same held-out 2024 season, each against a
baseline that makes its number interpretable:

    compound  5-class  vs. always predicting the most common compound
    laptime   regression  vs. predicting the stint median (a delta of zero)
    outcome   3-class  vs. always predicting the most common outcome

Each head is scored only where its own mask says the target exists -- the
compound head, for instance, is meaningful only where a stop is actually coming.

Usage
-----
    python -m src.evaluation.eval_mtl_heads
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, r2_score)

from src import config
from src.evaluation import plots
from src.evaluation.plots import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SEQUENTIAL, SURFACE

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402

DPI = 200
DISPLAY = "Model 3 — Multi-task + cross-driver attention"


def load_model_and_grids():
    from src.models.mtl_architecture import build_mtl_attention
    from src.models.train_mtl import grid_arrays, load_grids

    with open(config.METRICS_DIR / "mtl_attn_metrics.json") as fh:
        stored = json.load(fh)
    seq, mtl, spec, X, n_train, n_val = load_grids()

    model = build_mtl_attention(
        seq_len=spec["seq_len"], n_features=spec["n_features"],
        max_drivers=spec["max_drivers"], n_compounds=spec["n_compound_classes"],
        n_outcomes=spec["n_outcome_classes"], pos_weight=stored["pos_weight"])
    model.load_weights(config.MODELS_DIR / "mtl_attn.weights.h5")

    x_te, y_te, _, m_te = grid_arrays(mtl, X, "test", spec)
    return model, x_te, y_te, m_te, spec


def flat(arr, mask):
    """Flatten a (grids, drivers, ...) array down to the rows the mask keeps."""
    sel = mask > 0
    return arr[sel]


def plot_cm(cm, labels, title, stem, note=None):
    n = len(labels)
    row = cm.sum(1, keepdims=True)
    pct = np.divide(cm, np.where(row == 0, 1, row)) * 100

    fig, ax = plt.subplots(figsize=(1.35 * n + 2.4, 1.25 * n + 2.0))
    im = ax.imshow(pct, cmap=SEQUENTIAL, vmin=0, vmax=100)
    ax.set_xticks(range(n), labels=labels, rotation=20, ha="right")
    ax.set_yticks(range(n), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            colour = SURFACE if pct[i, j] > 55 else INK_PRIMARY
            ax.text(j, i - 0.09, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=colour)
            ax.text(j, i + 0.17, f"{pct[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color=colour)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of actual class", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    if note:
        ax.annotate(note, xy=(0, -0.26), xycoords="axes fraction", fontsize=8,
                    color=INK_MUTED, va="top")
    path = config.FIGURES_DIR / f"mtl_head_{stem}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> int:
    config.ensure_dirs()
    config.set_seed()
    plots.apply_style()

    model, x_te, y_te, m_te, spec = load_model_and_grids()
    pred = model.predict(x_te, batch_size=64, verbose=0)
    written, results = [], {}

    print(f"{DISPLAY}\nheld-out 2024 season — {int(m_te.sum()):,} driver-laps\n")

    # ------------------------------------------------- compound (5-class) --
    mask = y_te["compound_mask"] * m_te
    y_true = flat(y_te["compound"], mask).astype(int)
    y_pred = flat(pred["compound"].argmax(-1), mask).astype(int)
    names = spec["compounds"]
    present = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    majority = np.bincount(y_true, minlength=len(names)).argmax()
    base = accuracy_score(y_true, np.full_like(y_true, majority))
    cm = confusion_matrix(y_true, y_pred, labels=present)
    results["compound"] = {"n": int(len(y_true)), "accuracy": float(acc),
                           "macro_f1": float(macro), "baseline_accuracy": float(base),
                           "baseline_class": names[majority],
                           "confusion_matrix": cm.tolist(),
                           "classes": [names[i] for i in present]}
    print(f"COMPOUND  (5-class, {len(y_true):,} rows where a stop is coming)")
    print(f"  accuracy {acc:.4f}   macro-F1 {macro:.4f}   "
          f"majority baseline {base:.4f} (always {names[majority]})")
    written.append(plot_cm(cm, [names[i] for i in present],
                           f"{DISPLAY}\nCompound head — next tyre fitted",
                           "compound",
                           note=f"Scored only on the {len(y_true):,} laps where a stop "
                                f"actually follows. Majority baseline {base:.1%}."))

    # ------------------------------------------------------ laptime (reg) --
    mask = y_te["laptime_mask"] * m_te
    t_std = flat(y_te["laptime"], mask)
    p_std = flat(pred["laptime"][..., 0], mask)
    # Back to seconds: the target is a delta to the stint median, standardised
    # on the training split.
    s, mu = spec["laptime_std"], spec["laptime_mean"]
    t_sec, p_sec = t_std * s + mu, p_std * s + mu
    mae = mean_absolute_error(t_sec, p_sec)
    rmse = float(np.sqrt(np.mean((t_sec - p_sec) ** 2)))
    r2 = r2_score(t_sec, p_sec)
    base_mae = mean_absolute_error(t_sec, np.full_like(t_sec, 0.0))
    results["laptime"] = {"n": int(len(t_sec)), "mae_s": float(mae), "rmse_s": rmse,
                          "r2": float(r2), "baseline_mae_s": float(base_mae)}
    print(f"\nLAPTIME   (regression, {len(t_sec):,} rows)")
    print(f"  MAE {mae:.3f} s   RMSE {rmse:.3f} s   R² {r2:.4f}   "
          f"stint-median baseline MAE {base_mae:.3f} s")

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    lim = 8
    idx = np.random.default_rng(0).choice(len(t_sec), size=min(6000, len(t_sec)), replace=False)
    ax.scatter(t_sec[idx], p_sec[idx], s=5, alpha=0.18, color="#1baf7a", edgecolors="none")
    ax.plot([-lim, lim], [-lim, lim], color="#c3c2b7", lw=1, ls="--", zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual next-lap pace (s vs. stint median)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{DISPLAY}\nLap-time head — next-lap pace")
    ax.annotate(f"MAE {mae:.2f} s   R² {r2:.3f}", xy=(0.04, 0.93),
                xycoords="axes fraction", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax.set_axisbelow(True)
    p = config.FIGURES_DIR / "mtl_head_laptime.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    written.append(str(p))

    # ------------------------------------------------------ outcome (3-cl) --
    mask = y_te["outcome_mask"] * m_te
    y_true = flat(y_te["outcome"], mask).astype(int)
    y_pred = flat(pred["outcome"].argmax(-1), mask).astype(int)
    names = ["Podium", "Points", "No points"]
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    majority = np.bincount(y_true, minlength=3).argmax()
    base = accuracy_score(y_true, np.full_like(y_true, majority))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    results["outcome"] = {"n": int(len(y_true)), "accuracy": float(acc),
                          "macro_f1": float(macro), "baseline_accuracy": float(base),
                          "baseline_class": names[majority],
                          "confusion_matrix": cm.tolist(), "classes": names}
    print(f"\nOUTCOME   (3-class, {len(y_true):,} rows)")
    print(f"  accuracy {acc:.4f}   macro-F1 {macro:.4f}   "
          f"majority baseline {base:.4f} (always {names[majority]})")
    written.append(plot_cm(cm, names, f"{DISPLAY}\nOutcome head — final race result",
                           "outcome",
                           note=f"Predicted from each lap's state. Majority baseline {base:.1%}."))

    out = config.METRICS_DIR / "mtl_auxiliary_heads.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nArtefacts written")
    print(f"  metrics  {out}")
    for w in written:
        print(f"  figure   {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
