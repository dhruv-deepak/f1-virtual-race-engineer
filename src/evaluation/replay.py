"""
Phase 6 -- historical race replay.

The project's eighth objective is to evaluate the model's recommendations
against the strategy calls real race engineers actually made. The test set is
the whole 2024 season, which no model saw during training, so replaying it is a
direct comparison of machine and human decisions on the same races.

What a replay measures
----------------------
Per-lap metrics answer "how often is the model right about a lap". A race
engineer asks a different question: *of the stops that happened, how many did we
see coming, and how early?* So the replay scores **per stop**, not per lap:

* **anticipated** -- the model's probability crossed its threshold on at least
  one of the ``horizon`` laps before the stop.
* **lead time** -- how many laps before the stop it first crossed.
* **false alarms** -- crossings that no stop followed, counted per driver-race,
  which is the number a strategist would actually feel.

Consecutive crossings before one stop are a single alarm, not three. Counting
each lap separately would badly overstate how noisy the system is, because the
label itself spans a three-lap window.

Usage
-----
    python -m src.evaluation.replay
    python -m src.evaluation.replay --race 2024_11 --driver VER
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from src import config
from src.evaluation import plots
from src.evaluation.compare import ORDER, SERIES, SHORT
from src.evaluation.plots import AXIS, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

DEFAULT_RACE = "2024_11"      # Austrian GP: 71 laps, 20 drivers, busy strategy
DEFAULT_DRIVER = "VER"
DPI = 200

# One-hue ramp for the probability heat strip: probability is a magnitude.
HEAT = LinearSegmentedColormap.from_list(
    "f1_heat", ["#f8fafc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def load_test_frame() -> tuple[pd.DataFrame, int]:
    """Test-split rows with each model's probability attached."""
    meta = pd.read_parquet(config.DATA_PROCESSED / "meta.parquet")
    test = meta[meta["split"] == "test"].reset_index(drop=True).copy()

    seq = np.load(config.DATA_PROCESSED / "sequences.npz")
    y_test = seq["y_test"]
    if len(test) != len(y_test):
        raise ValueError(f"meta holds {len(test)} test rows but y_test has {len(y_test)}")
    test["y_pit"] = y_test

    thresholds = {}
    for name in ORDER:
        probs = np.load(config.METRICS_DIR / f"{name}_probs.npz")["test"]
        if len(probs) != len(test):
            raise ValueError(f"{name}: {len(probs)} probabilities for {len(test)} rows")
        test[name] = probs
        with open(config.METRICS_DIR / f"{name}_metrics.json") as fh:
            thresholds[name] = json.load(fh)["tuned_threshold"]

    with open(config.DATA_PROCESSED / "feature_names.json") as fh:
        horizon = json.load(fh)["horizon"]
    test.attrs["thresholds"] = thresholds
    return test, horizon


def actual_stop_laps(df: pd.DataFrame, horizon: int) -> list[float]:
    """
    Recover the laps a driver actually pitted, from the label itself.

    y_pit marks the laps *before* a stop, so a stop sits at the lap right after
    a run of ones ends. Deriving it this way rather than re-reading the raw data
    guarantees the replay is scored against exactly the same events the models
    were trained and evaluated on.
    """
    df = df.sort_values("LapNumber")
    laps = df["LapNumber"].to_numpy()
    y = df["y_pit"].to_numpy()
    stops = []
    for i in range(len(y)):
        if y[i] == 1 and (i + 1 == len(y) or y[i + 1] == 0):
            stops.append(laps[i] + 1)        # the lap the stop lands on
    return stops


def score_driver(df: pd.DataFrame, model: str, threshold: float, horizon: int) -> dict:
    """Per-stop anticipation and alarm counts for one driver in one race."""
    df = df.sort_values("LapNumber")
    laps = df["LapNumber"].to_numpy()
    prob = df[model].to_numpy()
    fired = prob >= threshold
    stops = actual_stop_laps(df, horizon)

    anticipated, lead_times = 0, []
    for stop in stops:
        window = (laps >= stop - horizon) & (laps <= stop - 1)
        if fired[window].any():
            anticipated += 1
            first = laps[window][fired[window]][0]
            lead_times.append(stop - first)

    # Group consecutive firing laps into single alarms.
    alarms, true_alarms = 0, 0
    i = 0
    while i < len(fired):
        if not fired[i]:
            i += 1
            continue
        start = i
        while i < len(fired) and fired[i]:
            i += 1
        alarms += 1
        block = laps[start:i]
        if any(any(stop - horizon <= lap <= stop - 1 for lap in block) for stop in stops):
            true_alarms += 1

    return {"stops": len(stops), "anticipated": anticipated,
            "lead_times": lead_times, "alarms": alarms,
            "false_alarms": alarms - true_alarms}


def season_summary(test: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Aggregate the replay across every driver of every 2024 race."""
    thresholds = test.attrs["thresholds"]
    rows = []
    for name in ORDER:
        total = {"stops": 0, "anticipated": 0, "alarms": 0, "false_alarms": 0}
        leads, driver_races = [], 0
        for _, grp in test.groupby(["RaceId", "Driver"], sort=False):
            s = score_driver(grp, name, thresholds[name], horizon)
            for k in total:
                total[k] += s[k]
            leads += s["lead_times"]
            driver_races += 1
        rows.append({
            "model": SHORT[name],
            "threshold": round(thresholds[name], 3),
            "stops_in_season": total["stops"],
            "anticipated": total["anticipated"],
            "anticipated_pct": round(100 * total["anticipated"] / max(total["stops"], 1), 1),
            "median_lead_laps": float(np.median(leads)) if leads else float("nan"),
            "false_alarms": total["false_alarms"],
            "false_alarms_per_driver_race": round(total["false_alarms"] / max(driver_races, 1), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_driver_replay(test, race, driver, horizon, event_name):
    """One driver's race: every model's probability against the real stops."""
    df = test[(test["RaceId"] == race) & (test["Driver"] == driver)].sort_values("LapNumber")
    if df.empty:
        raise ValueError(f"no test rows for {driver} at {race}")
    thresholds = test.attrs["thresholds"]
    stops = actual_stop_laps(df, horizon)
    laps = df["LapNumber"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    for stop in stops:
        ax.axvspan(stop - horizon, stop, color="#e1e1ea", zorder=0)
        ax.axvline(stop, color=INK_PRIMARY, lw=1.6, zorder=4)
        ax.annotate(f"actual stop\nlap {int(stop)}", xy=(stop, 1.02),
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=8, color=INK_PRIMARY)

    for name in ORDER:
        ax.plot(laps, df[name].to_numpy(), color=SERIES[name], label=SHORT[name], lw=1.8)
        ax.axhline(thresholds[name], color=SERIES[name], lw=0.9, ls=":", alpha=0.65, zorder=1)

    ax.set_xlabel("Lap")
    ax.set_ylabel("P(pit within 3 laps)")
    # Extra pad so the per-stop callouts above the axes clear the title.
    ax.set_title(f"Race replay — {driver}, {event_name} 2024", pad=34)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(laps.min(), laps.max())
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.annotate("shaded band = the 3 laps a correct call had to fire in;  "
                "dotted lines = each model's tuned threshold",
                xy=(0, -0.20), xycoords="axes fraction", fontsize=8, color=INK_MUTED,
                va="top")
    path = config.FIGURES_DIR / f"replay_{race}_{driver}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_race_grid(test, race, model, horizon, event_name):
    """The whole field: a probability heat strip per driver with real stops marked."""
    df = test[test["RaceId"] == race]
    drivers = sorted(df["Driver"].unique())
    lap_min, lap_max = int(df["LapNumber"].min()), int(df["LapNumber"].max())
    laps = np.arange(lap_min, lap_max + 1)

    heat = np.full((len(drivers), len(laps)), np.nan)
    marks = []
    for r, drv in enumerate(drivers):
        d = df[df["Driver"] == drv].sort_values("LapNumber")
        idx = (d["LapNumber"].to_numpy() - lap_min).astype(int)
        heat[r, idx] = d[model].to_numpy()
        for stop in actual_stop_laps(d, horizon):
            if lap_min <= stop <= lap_max:
                marks.append((stop - lap_min, r))

    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    im = ax.imshow(heat, aspect="auto", cmap=HEAT, vmin=0, vmax=1,
                   extent=[lap_min - 0.5, lap_max + 0.5, len(drivers) - 0.5, -0.5])
    if marks:
        xs = [lap_min + m[0] for m in marks]
        ys = [m[1] for m in marks]
        ax.scatter(xs, ys, marker="|", s=170, linewidths=2.1, color="#c41e3a",
                   zorder=3, label="actual pit stop")
    ax.set_yticks(range(len(drivers)), labels=drivers, fontsize=8)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Driver")
    ax.set_title(f"{SHORT[model]} — predicted pit probability across the field, "
                 f"{event_name} 2024")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=SURFACE)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_label("P(pit within 3 laps)", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    path = config.FIGURES_DIR / f"replay_grid_{race}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay 2024 races against the models.")
    parser.add_argument("--race", default=DEFAULT_RACE)
    parser.add_argument("--driver", default=DEFAULT_DRIVER)
    args = parser.parse_args()

    config.ensure_dirs()
    plots.apply_style()

    test, horizon = load_test_frame()
    if args.race not in set(test["RaceId"]):
        raise SystemExit(f"{args.race} is not in the test split")
    event = test.loc[test["RaceId"] == args.race, "EventName"].iloc[0]

    print(f"Replaying the {test['RaceId'].nunique()} races of the 2024 season "
          f"({len(test):,} driver-laps)\n")
    summary = season_summary(test, horizon)
    print(summary.to_string(index=False))
    out_csv = config.METRICS_DIR / "race_replay_summary.csv"
    summary.to_csv(out_csv, index=False)

    written = [
        plot_driver_replay(test, args.race, args.driver, horizon, event),
        plot_race_grid(test, args.race, ORDER[0], horizon, event),
    ]

    best = summary.sort_values("anticipated_pct", ascending=False).iloc[0]
    print(f"\nbest anticipation: {best['model']} saw {best['anticipated']} of "
          f"{best['stops_in_season']} stops coming ({best['anticipated_pct']}%), "
          f"median {best['median_lead_laps']:.0f} laps early, "
          f"{best['false_alarms_per_driver_race']} false alarms per driver-race")

    print("\nArtefacts written")
    for w in written:
        print(f"  figure  {w}")
    print(f"  table   {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
