"""
Phase 7 -- animated race replay for the slide deck.

Renders the field-wide probability heat strip filling in lap by lap and writes
it as an animated GIF. PowerPoint plays a GIF inline with no plugin and no
clicking, so this is the demo that cannot fail in front of a panel, with the
interactive page as the richer version when the laptop cooperates.

Usage
-----
    python -m src.evaluation.build_replay_gif
    python -m src.evaluation.build_replay_gif --race 2024_11 --fps 9
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
from PIL import Image

from src import config
from src.evaluation import plots
from src.evaluation.compare import ORDER, SHORT
from src.evaluation.replay import HEAT, actual_stop_laps, load_test_frame

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_RACE = "2024_11"
DPI = 88


def build_matrix(test, race, horizon):
    df = test[test["RaceId"] == race]
    drivers = sorted(df["Driver"].unique())
    lap_min, lap_max = int(df["LapNumber"].min()), int(df["LapNumber"].max())
    laps = np.arange(lap_min, lap_max + 1)

    heat = np.full((len(drivers), len(laps)), np.nan)
    stops = []
    for r, drv in enumerate(drivers):
        d = df[df["Driver"] == drv].sort_values("LapNumber")
        idx = (d["LapNumber"].to_numpy() - lap_min).astype(int)
        heat[r, idx] = d[ORDER[0]].to_numpy()
        stops += [(s, r) for s in actual_stop_laps(d, horizon) if lap_min <= s <= lap_max]
    return drivers, laps, heat, stops, str(df["EventName"].iloc[0])


def render_frame(drivers, laps, heat, stops, event, upto):
    """One frame: everything up to `upto` revealed, the rest left blank."""
    shown = heat.copy()
    shown[:, upto + 1:] = np.nan
    lap_min, lap_max = laps[0], laps[-1]

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.imshow(shown, aspect="auto", cmap=HEAT, vmin=0, vmax=1,
              extent=[lap_min - 0.5, lap_max + 0.5, len(drivers) - 0.5, -0.5])

    visible = [(s, r) for s, r in stops if s <= laps[upto]]
    if visible:
        ax.scatter([s for s, _ in visible], [r for _, r in visible], marker="|",
                   s=170, linewidths=2.1, color="#c41e3a", zorder=3)

    # The playhead makes the direction of travel obvious at a glance.
    ax.axvline(laps[upto], color="#15151e", lw=1.1, alpha=0.45, zorder=4)
    ax.set_yticks(range(len(drivers)), labels=drivers, fontsize=8)
    ax.set_xlim(lap_min - 0.5, lap_max + 0.5)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Driver")
    ax.set_title(f"{SHORT[ORDER[0]]} — predicted pit probability, {event} 2024")
    ax.annotate(f"LAP {int(laps[upto])}", xy=(0.995, 1.012), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=12, fontweight="bold", color="#c41e3a")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                           fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
    plt.close(fig)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Animate a race replay as a GIF.")
    parser.add_argument("--race", default=DEFAULT_RACE)
    parser.add_argument("--fps", type=int, default=9)
    parser.add_argument("--hold", type=int, default=14,
                        help="extra frames held on the finished race before looping")
    args = parser.parse_args()

    config.ensure_dirs()
    plots.apply_style()
    plt.rcParams["figure.dpi"] = DPI

    test, horizon = load_test_frame()
    if args.race not in set(test["RaceId"]):
        raise SystemExit(f"{args.race} is not in the test split")

    drivers, laps, heat, stops, event = build_matrix(test, args.race, horizon)
    print(f"{event} 2024 — {len(drivers)} drivers, laps {laps[0]}–{laps[-1]}")

    frames = [render_frame(drivers, laps, heat, stops, event, i) for i in range(len(laps))]
    frames += [frames[-1]] * args.hold        # pause on the complete race before looping

    out = config.FIGURES_DIR / f"replay_animated_{args.race}.gif"
    # Palette quantisation keeps the file small enough to embed in a deck; the
    # heat ramp is a single hue, so 128 colours is more than it needs.
    frames = [f.quantize(colors=128, method=Image.MEDIANCUT) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)

    print(f"{out}")
    print(f"  {len(frames)} frames at {args.fps} fps, {out.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
