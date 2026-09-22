"""
Phase 1 -- FastF1 data acquisition.

Downloads every Formula 1 race session for the configured seasons and writes one
tidy parquet file per race into ``data/raw/``. Each file is a driver-lap table:
one row per driver per lap, already joined with weather, per-lap aggregated car
telemetry, and that driver's final race result.

Why telemetry is aggregated here instead of stored raw
------------------------------------------------------
FastF1 samples car data at roughly 3.7 Hz. Across ~90 races x ~20 drivers that
is tens of gigabytes of raw samples -- unusable on a laptop, and far more
resolution than a lap-level strategy model needs. We therefore reduce each lap's
telemetry trace to eight summary channels (mean/max speed, throttle, full
throttle %, brake %, RPM, gear) at download time. The reduction is vectorised
and costs roughly 0.25s per race.

Usage
-----
    python -m src.data.download                    # all configured seasons
    python -m src.data.download --seasons 2023     # one season
    python -m src.data.download --limit 2          # smoke test: 2 races/season
    python -m src.data.download --force            # re-download existing files
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import warnings

import fastf1
import pandas as pd

from src import config

warnings.filterwarnings("ignore")
# FastF1 is extremely chatty on INFO; we only want to hear about real problems.
logging.getLogger("fastf1").setLevel(logging.ERROR)

# Lap columns FastF1 returns as timedeltas. Parquet round-trips these
# inconsistently across engines, so we store them as plain float seconds.
TIMEDELTA_COLS = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]

# Session-clock columns, also timedeltas, converted the same way.
SESSION_TIME_COLS = ["Time", "LapStartTime", "PitInTime", "PitOutTime"]

# Driver-level context taken from the results table.
RESULT_COLS = ["DriverNumber", "GridPosition", "Position", "Points", "Status", "ClassifiedPosition"]

# Weather channels aligned onto each lap.
WEATHER_COLS = ["Time", "AirTemp", "TrackTemp", "Humidity", "Pressure",
                "WindSpeed", "WindDirection", "Rainfall"]

# Columns dropped at the raw stage: redundant with what we keep, or unusable.
DROP_COLS = ["LapStartDate", "Sector1SessionTime", "Sector2SessionTime",
             "Sector3SessionTime", "DeletedReason"]


def _slug(text: str) -> str:
    """Turn an event name into a safe filename fragment."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def aggregate_telemetry(session) -> pd.DataFrame:
    """
    Reduce each driver's raw car-data trace to one summary row per lap.

    Telemetry samples carry a session ``Time``; laps carry ``LapStartTime`` and
    an end ``Time``. We bin the samples into laps with ``pd.cut`` using the lap
    start times as bin edges, which is a single vectorised pass per driver
    rather than a per-lap slice of the telemetry frame.
    """
    laps = session.laps
    frames = []

    for drv_num, tel in session.car_data.items():
        drv_laps = laps[laps["DriverNumber"] == drv_num]
        if drv_laps.empty or tel.empty:
            continue
        drv_laps = drv_laps.sort_values("LapNumber")

        # Bin edges: every lap start, plus the final lap's end time.
        edges = pd.concat([drv_laps["LapStartTime"], drv_laps["Time"].iloc[[-1]]]).dropna()
        if len(edges) < 2:
            continue

        lap_index = pd.cut(
            tel["Time"],
            bins=edges,
            labels=drv_laps["LapNumber"].iloc[: len(edges) - 1],
            right=False,
            ordered=False,
        )
        grouped = (
            tel.assign(_lap=lap_index)
            .dropna(subset=["_lap"])
            .groupby("_lap", observed=True)
        )

        agg = pd.DataFrame(
            {
                "tel_speed_mean": grouped["Speed"].mean(),
                "tel_speed_max": grouped["Speed"].max(),
                "tel_throttle_mean": grouped["Throttle"].mean(),
                # Share of the lap spent at full throttle -- a compact proxy for
                # how much of the circuit is flat out under current conditions.
                "tel_full_throttle_pct": grouped["Throttle"].apply(lambda x: (x > 95).mean() * 100),
                "tel_brake_pct": grouped["Brake"].apply(lambda x: x.astype(bool).mean() * 100),
                "tel_rpm_mean": grouped["RPM"].mean(),
                "tel_gear_mean": grouped["nGear"].mean(),
                "tel_n_samples": grouped["Speed"].size(),
            }
        )
        agg.index.name = "LapNumber"
        agg = agg.reset_index()
        agg["DriverNumber"] = drv_num
        frames.append(agg)

    if not frames:
        return pd.DataFrame(columns=["LapNumber", "DriverNumber"])
    return pd.concat(frames, ignore_index=True)


def build_race_table(session, year: int, rnd: int, event) -> pd.DataFrame:
    """Join laps + weather + telemetry + results into a single driver-lap table."""
    df = session.laps.copy()
    if df.empty:
        raise ValueError("no lap data returned for this session")

    # --- weather: nearest reading at or before the end of each lap -----------
    weather = session.weather_data
    if weather is not None and not weather.empty:
        df = df.sort_values("Time")
        w = weather.sort_values("Time")[[c for c in WEATHER_COLS if c in weather.columns]]
        df = pd.merge_asof(df, w, on="Time", direction="backward")

    # --- per-lap telemetry summary -------------------------------------------
    tel = aggregate_telemetry(session)
    if not tel.empty:
        df = df.merge(tel, on=["DriverNumber", "LapNumber"], how="left")

    # --- final result per driver ---------------------------------------------
    results = session.results
    if results is not None and not results.empty:
        res = results[[c for c in RESULT_COLS if c in results.columns]].copy()
        res = res.rename(columns={"Position": "FinalPosition"})
        df = df.merge(res, on="DriverNumber", how="left", suffixes=("", "_res"))

    # --- pit flags, then timedelta -> float seconds ---------------------------
    # The boolean flags are what the label builder needs; the raw timestamps are
    # only kept for ordering.
    df["IsPitInLap"] = df["PitInTime"].notna() if "PitInTime" in df.columns else False
    df["IsPitOutLap"] = df["PitOutTime"].notna() if "PitOutTime" in df.columns else False

    for col in TIMEDELTA_COLS + SESSION_TIME_COLS:
        if col in df.columns:
            df[col] = df[col].dt.total_seconds()

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # --- race-level context ---------------------------------------------------
    df["Season"] = year
    df["Round"] = rnd
    df["EventName"] = event["EventName"]
    df["Location"] = event["Location"]
    df["Country"] = event["Country"]
    df["TotalLaps"] = session.total_laps
    # Stable race identifier -- the unit our train/val/test split works on.
    df["RaceId"] = f"{year}_{rnd:02d}"

    return df


def download_season(year: int, force: bool = False, limit: int | None = None) -> list[str]:
    """Download every race of one season. Returns a list of failure messages."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if limit:
        schedule = schedule.head(limit)

    failures = []
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        name = event["EventName"]
        out_path = config.DATA_RAW / f"{year}_{rnd:02d}_{_slug(name)}.parquet"

        if out_path.exists() and not force:
            print(f"  [skip] {year} R{rnd:02d} {name}")
            continue

        t0 = time.time()
        try:
            session = fastf1.get_session(year, rnd, config.SESSION_TYPE)
            session.load(laps=True, telemetry=True, weather=True, messages=False)
            df = build_race_table(session, year, rnd, event)
            df.to_parquet(out_path, index=False)
            print(f"  [ok]   {year} R{rnd:02d} {name:<30.30} "
                  f"{len(df):>5} laps  {df['Driver'].nunique():>2} drivers  "
                  f"{time.time() - t0:>5.1f}s")
        except Exception as exc:
            # A cancelled or unavailable session must not abort the whole sweep.
            msg = f"{year} R{rnd:02d} {name}: {type(exc).__name__}: {exc}"
            print(f"  [FAIL] {msg}")
            failures.append(msg)

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Download F1 race data via FastF1.")
    parser.add_argument("--seasons", type=int, nargs="+", default=config.SEASONS,
                        help=f"seasons to download (default: {config.SEASONS})")
    parser.add_argument("--force", action="store_true",
                        help="re-download races that already exist on disk")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N races of each season")
    args = parser.parse_args()

    config.ensure_dirs()
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))

    all_failures = []
    t_start = time.time()
    for year in args.seasons:
        print(f"\n=== Season {year} ===")
        all_failures += download_season(year, force=args.force, limit=args.limit)

    files = sorted(config.DATA_RAW.glob("*.parquet"))
    total_rows = sum(len(pd.read_parquet(f, columns=["LapNumber"])) for f in files)
    print("\n" + "=" * 64)
    print(f"Races on disk : {len(files)}")
    print(f"Driver-laps   : {total_rows:,}")
    print(f"Elapsed       : {time.time() - t_start:.0f}s")
    if all_failures:
        print(f"\nFailures ({len(all_failures)}):")
        for f in all_failures:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
