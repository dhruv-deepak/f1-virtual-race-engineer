"""
Phase 2 -- Feature engineering and sequence dataset construction.

Turns the per-race parquet files in ``data/raw/`` into the model-ready tensors
that all three architectures consume:

    X : float32 (n_samples, SEQ_LEN, n_features)   last 10 laps of context
    y : int8    (n_samples,)                       pits within next 3 laps?

Design decisions worth knowing
------------------------------
1. **Label horizon.** ``y = 1`` when the driver pits on any of laps
   ``t+1 .. t+PIT_HORIZON``. Lap ``t`` itself is excluded: a pit stop on lap
   ``t`` is already visible in the input window, so including it would be label
   leakage. Widening from 1 lap to 3 also lifts the positive rate from ~5% to a
   trainable ~13%.

2. **Split by race, not by row.** Laps from the same race are highly
   correlated, so a random row split would put near-duplicate laps in both
   train and test and inflate accuracy. Seasons 2021-22 train, 2023 validates,
   2024 tests.

3. **Scaler fit on train only.** Fitting the StandardScaler on all data would
   leak test-set statistics into training.

4. **Competitor features.** Undercut pressure is a cross-driver phenomenon, so
   each lap also carries field-relative context: gap to leader, intervals to the
   cars ahead and behind, how much of the field has already pitted, and this
   driver's tyre age relative to the field. These are the hand-built precursor
   to the learned cross-driver attention in Model 3.

Usage
-----
    python -m src.features.build_dataset
    python -m src.features.build_dataset --seq-len 15 --horizon 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config

warnings.filterwarnings("ignore")

# FastF1 track-status codes can be concatenated on one lap, e.g. "41" means the
# lap saw both a safety car (4) and green flag (1).
TRACK_STATUS_FLAGS = {"2": "ts_yellow", "4": "ts_safety_car", "5": "ts_red", "6": "ts_vsc"}

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

WEATHER_FEATURES = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]

TELEMETRY_FEATURES = [
    "tel_speed_mean", "tel_speed_max", "tel_throttle_mean",
    "tel_full_throttle_pct", "tel_brake_pct", "tel_rpm_mean", "tel_gear_mean",
]

SPEED_TRAP_FEATURES = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]

# Identify each sample so Phase 6 can replay a specific race lap by lap.
META_COLS = ["RaceId", "Season", "Round", "EventName", "Driver", "LapNumber"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Concatenate every downloaded race into one driver-lap frame."""
    files = sorted(config.DATA_RAW.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files in {config.DATA_RAW}. Run `python -m src.data.download` first."
        )
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df.sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable laps and driver-races too short to form a single window."""
    before = len(df)

    df = df[df["LapNumber"].notna() & df["Driver"].notna()]
    # A lap with no recorded time carries no timing signal at all. These are rare
    # (typically the lap on which a car retires) and cannot be imputed sensibly.
    df = df[df["LapTime"].notna()]
    df["Compound"] = df["Compound"].fillna("UNKNOWN").str.upper()

    # A driver needs SEQ_LEN laps of history plus PIT_HORIZON laps of future
    # before a single labelled sample can be built.
    counts = df.groupby(["RaceId", "Driver"])["LapNumber"].transform("size")
    df = df[counts >= config.MIN_LAPS_PER_DRIVER]

    print(f"  cleaned: {before:,} -> {len(df):,} driver-laps "
          f"({before - len(df):,} dropped)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _rolling_slope(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Vectorised rolling least-squares slope of y on x.

    Used as the tyre-degradation proxy: the slope of lap time against tyre age
    within the current stint. A steepening positive slope means the tyre is
    falling away, which is precisely the signal a race engineer watches before
    calling a driver in. Computed as cov(x, y) / var(x) so it stays a few vector
    operations instead of a per-window polyfit.
    """
    mx = x.rolling(window, min_periods=window).mean()
    my = y.rolling(window, min_periods=window).mean()
    cov = (x * y).rolling(window, min_periods=window).mean() - mx * my
    var = (x * x).rolling(window, min_periods=window).mean() - mx * mx
    return cov / var.replace(0, np.nan)


def add_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver, per-race sequential features (timing, tyre, stint state)."""
    g = df.groupby(["RaceId", "Driver"], sort=False)
    gs = df.groupby(["RaceId", "Driver", "Stint"], sort=False)

    df["lap_time"] = df["LapTime"]
    # Pace relative to this stint's own typical lap removes circuit and car
    # effects, leaving the degradation trend the model actually needs.
    df["lap_time_delta_stint"] = df["LapTime"] - gs["LapTime"].transform("median")
    df["lap_time_diff"] = g["LapTime"].diff()
    df["lap_time_roll3_mean"] = g["LapTime"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["lap_time_roll3_std"] = g["LapTime"].transform(lambda s: s.rolling(3, min_periods=2).std())

    # Degradation slope within the current stint only -- it must reset at a stop.
    # group_keys=False keeps the original index, so the result aligns directly.
    df["deg_slope5"] = df.groupby(
        ["RaceId", "Driver", "Stint"], sort=False, group_keys=False
    ).apply(lambda d: _rolling_slope(d["TyreLife"], d["LapTime"], 5))

    df["tyre_life"] = df["TyreLife"]
    df["stint"] = df["Stint"]
    df["is_fresh_tyre"] = df["FreshTyre"].astype(float)

    df["is_pit_in_lap"] = df["IsPitInLap"].astype(float)
    df["is_pit_out_lap"] = df["IsPitOutLap"].astype(float)
    # Laps run on the current set. FastF1's TyreLife already counts this and
    # correctly carries over when a used set is refitted, so it is a cleaner
    # measure than counting laps since the last pit flag.
    df["laps_since_pit"] = df["TyreLife"]

    df["position"] = df["Position"]
    df["grid_position"] = df["GridPosition"]
    df["position_change"] = df["GridPosition"] - df["Position"]

    df["lap_number"] = df["LapNumber"]
    df["lap_frac"] = df["LapNumber"] / df["TotalLaps"].replace(0, np.nan)
    df["laps_remaining"] = df["TotalLaps"] - df["LapNumber"]

    df["sector1"] = df["Sector1Time"]
    df["sector2"] = df["Sector2Time"]
    df["sector3"] = df["Sector3Time"]

    return df


def add_track_status(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the concatenated TrackStatus string into binary flags."""
    status = df["TrackStatus"].fillna("").astype(str)
    for code, name in TRACK_STATUS_FLAGS.items():
        df[name] = status.str.contains(code, regex=False).astype(float)
    return df


def add_compound_onehot(df: pd.DataFrame) -> pd.DataFrame:
    for c in COMPOUNDS:
        df[f"compound_{c.lower()}"] = (df["Compound"] == c).astype(float)
    return df


def add_competitor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Field-relative context computed across all drivers on the same lap.

    ``Time`` is cumulative session time at the end of the lap, so differencing it
    between drivers on the same lap gives their on-track gap in seconds.
    """
    by_lap = df.groupby(["RaceId", "LapNumber"], sort=False)

    df["gap_to_leader"] = df["Time"] - by_lap["Time"].transform("min")
    # Lapped traffic produces enormous gaps that would dominate the scaler.
    df["gap_to_leader"] = df["gap_to_leader"].clip(upper=180)

    ordered = df.sort_values(["RaceId", "LapNumber", "Position"])
    interval = ordered.groupby(["RaceId", "LapNumber"], sort=False)["Time"].diff()
    df["interval_ahead"] = interval.reindex(df.index).clip(upper=60).fillna(0)
    df["interval_behind"] = (
        ordered.groupby(["RaceId", "LapNumber"], sort=False)["Time"]
        .diff(-1).abs().reindex(df.index).clip(upper=60).fillna(0)
    )

    # How much of the field is stopping right now, and how old this driver's
    # tyres are relative to everyone else -- the core undercut/overcut signals.
    df["n_pitted_this_lap"] = by_lap["is_pit_in_lap"].transform("sum")
    df["field_mean_tyre_life"] = by_lap["tyre_life"].transform("mean")
    df["tyre_life_vs_field"] = df["tyre_life"] - df["field_mean_tyre_life"]

    stops_so_far = df.groupby(["RaceId", "Driver"], sort=False)["is_pit_in_lap"].cumsum()
    df["stops_so_far"] = stops_so_far
    df["frac_field_pitted"] = (
        df.assign(_any=(stops_so_far > 0).astype(float))
        .groupby(["RaceId", "LapNumber"], sort=False)["_any"].transform("mean")
    )

    return df


def add_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    y = 1 if the driver pits on any of laps t+1 .. t+horizon.

    Implemented as a *forward* rolling sum on the shifted pit flag, so lap t's
    own stop never enters its own label.
    """
    def _fwd(s: pd.Series) -> pd.Series:
        return s.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).sum().iloc[::-1]

    df["y_pit"] = (
        df.groupby(["RaceId", "Driver"], sort=False)["is_pit_in_lap"]
        .transform(_fwd).fillna(0) > 0
    ).astype(np.int8)
    return df


def build_features(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    df = add_driver_features(df)
    df = add_track_status(df)
    df = add_compound_onehot(df)
    df = add_competitor_features(df)
    df = add_label(df, horizon)

    feature_names = (
        ["lap_time", "lap_time_delta_stint", "lap_time_diff",
         "lap_time_roll3_mean", "lap_time_roll3_std", "deg_slope5",
         "sector1", "sector2", "sector3"]
        + SPEED_TRAP_FEATURES
        + ["tyre_life", "stint", "is_fresh_tyre", "laps_since_pit",
           "is_pit_in_lap", "is_pit_out_lap", "stops_so_far"]
        + [f"compound_{c.lower()}" for c in COMPOUNDS]
        + ["position", "grid_position", "position_change",
           "lap_number", "lap_frac", "laps_remaining"]
        + list(TRACK_STATUS_FLAGS.values())
        + WEATHER_FEATURES
        + TELEMETRY_FEATURES
        + ["gap_to_leader", "interval_ahead", "interval_behind",
           "n_pitted_this_lap", "field_mean_tyre_life", "tyre_life_vs_field",
           "frac_field_pitted"]
    )

    df["Rainfall"] = df["Rainfall"].astype(float)
    for col in feature_names:
        if col not in df.columns:
            raise KeyError(f"feature '{col}' was never created")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remaining NaNs are warm-up artefacts of the rolling windows (first laps of
    # a stint). Fill forward within the driver, then zero.
    df[feature_names] = (
        df.groupby(["RaceId", "Driver"], sort=False)[feature_names]
        .ffill().fillna(0.0)
    )
    df[feature_names] = df[feature_names].replace([np.inf, -np.inf], 0.0)

    return df, feature_names


# ---------------------------------------------------------------------------
# Sequence windowing
# ---------------------------------------------------------------------------
def make_sequences(df: pd.DataFrame, feature_names: list[str], seq_len: int):
    """
    Slide a seq_len window over each driver-race.

    Sample i ends at lap t and contains laps t-seq_len+1 .. t; its label is the
    label attached to lap t. Windows never cross a driver or race boundary.
    """
    X_parts, y_parts, meta_parts = [], [], []

    for _, grp in df.groupby(["RaceId", "Driver"], sort=False):
        grp = grp.sort_values("LapNumber")
        n = len(grp)
        if n < seq_len:
            continue
        values = grp[feature_names].to_numpy(dtype=np.float32)
        labels = grp["y_pit"].to_numpy(dtype=np.int8)

        # Strided view: rows are consecutive windows, no data copied yet.
        windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
        windows = windows.transpose(0, 2, 1)          # (n_windows, seq_len, n_feat)

        X_parts.append(windows)
        y_parts.append(labels[seq_len - 1:])
        meta_parts.append(grp.iloc[seq_len - 1:][META_COLS])

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int8)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, y, meta


def split_by_season(meta: pd.DataFrame):
    """Boolean masks for the season-based train/val/test split."""
    season = meta["Season"].to_numpy()
    return (
        np.isin(season, config.TRAIN_SEASONS),
        np.isin(season, config.VAL_SEASONS),
        np.isin(season, config.TEST_SEASONS),
    )


def scale(X_tr, X_va, X_te, n_features):
    """Standardise features using statistics from the training split only."""
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, n_features))

    def _apply(X):
        if len(X) == 0:
            return X
        flat = scaler.transform(X.reshape(-1, n_features))
        return flat.reshape(X.shape).astype(np.float32)

    return _apply(X_tr), _apply(X_va), _apply(X_te), scaler


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the sequence dataset.")
    parser.add_argument("--seq-len", type=int, default=config.SEQ_LEN)
    parser.add_argument("--horizon", type=int, default=config.PIT_HORIZON)
    args = parser.parse_args()

    config.ensure_dirs()
    config.set_seed()

    print("Loading raw races ...")
    df = load_raw()
    print(f"  {df['RaceId'].nunique()} races, {len(df):,} driver-laps")

    print("Cleaning ...")
    df = clean(df)

    print(f"Engineering features (horizon={args.horizon} laps) ...")
    df, feature_names = build_features(df, args.horizon)
    print(f"  {len(feature_names)} features | positive rate "
          f"{100 * df['y_pit'].mean():.2f}%")

    print(f"Building sequences (seq_len={args.seq_len}) ...")
    X, y, meta = make_sequences(df, feature_names, args.seq_len)
    print(f"  X={X.shape}  y={y.shape}")

    tr, va, te = split_by_season(meta)
    print("Splitting by season ...")
    for name, mask in [("train", tr), ("val", va), ("test", te)]:
        if mask.sum():
            print(f"  {name:<5} {mask.sum():>7,} samples  "
                  f"{meta.loc[mask, 'RaceId'].nunique():>3} races  "
                  f"positives {100 * y[mask].mean():.2f}%")
        else:
            print(f"  {name:<5} EMPTY -- no races from seasons "
                  f"{getattr(config, name.upper() + '_SEASONS')} on disk")

    print("Scaling (fit on train only) ...")
    X_tr, X_va, X_te, scaler = scale(X[tr], X[va], X[te], len(feature_names))

    out = config.DATA_PROCESSED
    np.savez_compressed(out / "sequences.npz",
                        X_train=X_tr, y_train=y[tr],
                        X_val=X_va, y_val=y[va],
                        X_test=X_te, y_test=y[te])
    meta.assign(split=np.select([tr, va, te], ["train", "val", "test"], "unused")) \
        .to_parquet(out / "meta.parquet", index=False)
    with open(out / "feature_names.json", "w") as fh:
        json.dump({"features": feature_names,
                   "seq_len": args.seq_len,
                   "horizon": args.horizon,
                   "scaler_mean": scaler.mean_.tolist(),
                   "scaler_scale": scaler.scale_.tolist()}, fh, indent=2)

    print(f"\nSaved to {out}")
    print(f"  sequences.npz      X_train={X_tr.shape} X_val={X_va.shape} X_test={X_te.shape}")
    print(f"  meta.parquet       {len(meta):,} rows")
    print(f"  feature_names.json {len(feature_names)} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
