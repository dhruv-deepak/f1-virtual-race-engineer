"""
Phase 5 -- multi-task / cross-driver dataset layout.

Models 1 and 2 see one driver at a time: a sample is a single driver's 10-lap
window. Model 3 needs every driver who was on track at the same lap of the same
race in one sample, so attention can run *across* them.

This module produces that regrouping, plus the three extra task labels the
multi-task heads need.

Why it regroups instead of rebuilding
-------------------------------------
The feature tensors are taken straight from ``sequences.npz`` and only
*reindexed*. Model 3 therefore sees byte-identical features, the identical
train-only scaler and the identical season splits as Models 1 and 2 -- the sole
difference is how samples are grouped. Recomputing the features here would
reintroduce exactly the confound the Review 3 comparison is meant to exclude.

What a sample becomes
---------------------
    grid_idx : int32 (n_groups, MAX_DRIVERS)  row indices into X, -1 where empty
    mask     : bool  (n_groups, MAX_DRIVERS)  which slots hold a real driver

A training batch gathers ``X[grid_idx]`` to shape
``(batch, MAX_DRIVERS, SEQ_LEN, n_features)``. Padded slots are masked out of
both the attention and the loss.

The four task labels
--------------------
    y_pit       binary    pits within PIT_HORIZON laps      (the comparable head)
    y_compound  5-class   compound fitted at that next stop (defined only when
                          a stop is actually coming, so it carries its own mask)
    y_laptime   float     next lap's pace delta to its stint median
    y_outcome   3-class   podium / points / no points, constant per driver-race

Usage
-----
    python -m src.features.build_mtl_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from src import config
from src.features.build_dataset import COMPOUNDS, build_features, clean, load_raw

warnings.filterwarnings("ignore")

# A Formula 1 race starts at most 20 cars under current regulations. The build
# asserts the real maximum never exceeds this.
MAX_DRIVERS = 20

# Outcome classes, from the driver's final classified position.
OUTCOME_PODIUM, OUTCOME_POINTS, OUTCOME_NONE = 0, 1, 2
N_OUTCOME = 3

KEY = ["RaceId", "Driver", "LapNumber"]


# ---------------------------------------------------------------------------
# Auxiliary task labels
# ---------------------------------------------------------------------------
def add_aux_labels(df: pd.DataFrame) -> pd.DataFrame:
    """The three labels beyond y_pit that the multi-task heads learn from."""
    g = df.groupby(["RaceId", "Driver"], sort=False)

    # --- next compound -------------------------------------------------------
    # The compound fitted at the driver's *next* stop is the compound their next
    # stint starts on, so look up stint+1 rather than scanning forward lap by lap.
    stint_compound = (
        df.groupby(["RaceId", "Driver", "Stint"], sort=False)["Compound"]
        .first().rename("next_compound").reset_index()
    )
    stint_compound["Stint"] = stint_compound["Stint"] - 1      # shift back one stint
    df = df.merge(stint_compound, on=["RaceId", "Driver", "Stint"], how="left")

    code = {c: i for i, c in enumerate(COMPOUNDS)}
    df["y_compound"] = df["next_compound"].map(code).fillna(-1).astype(np.int8)
    # Only supervise this head where a stop is actually coming AND the compound
    # taken is known; everywhere else it is masked out of the loss entirely.
    df["y_compound_mask"] = ((df["y_pit"] == 1) & (df["y_compound"] >= 0)).astype(np.float32)
    df.loc[df["y_compound"] < 0, "y_compound"] = 0             # placeholder, masked

    # --- next lap pace -------------------------------------------------------
    # Delta to the stint median rather than raw seconds: circuit and car effects
    # cancel, leaving the degradation signal the head is meant to forecast.
    df["y_laptime"] = g["lap_time_delta_stint"].shift(-1)
    df["y_laptime_mask"] = df["y_laptime"].notna().astype(np.float32)
    df["y_laptime"] = df["y_laptime"].fillna(0.0).astype(np.float32)

    # --- race outcome --------------------------------------------------------
    pos = pd.to_numeric(df.get("FinalPosition"), errors="coerce")
    outcome = np.full(len(df), OUTCOME_NONE, dtype=np.int8)
    outcome[(pos >= 1) & (pos <= 3)] = OUTCOME_PODIUM
    outcome[(pos >= 4) & (pos <= 10)] = OUTCOME_POINTS
    df["y_outcome"] = outcome
    df["y_outcome_mask"] = pos.notna().astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def build_grid(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Group existing sample rows into (race, lap) grids of drivers.

    Returns row indices into X, padded with -1, plus a validity mask. Grids are
    never mixed across races or laps, and because every row of a grid comes from
    the same split, a grid belongs wholly to one split.
    """
    order = np.arange(len(meta), dtype=np.int32)
    grouped = (
        pd.DataFrame({"row": order,
                      "RaceId": meta["RaceId"].to_numpy(),
                      "LapNumber": meta["LapNumber"].to_numpy()})
        .groupby(["RaceId", "LapNumber"], sort=False)["row"].apply(list)
    )

    widest = max(len(v) for v in grouped)
    if widest > MAX_DRIVERS:
        raise ValueError(f"a (race, lap) group holds {widest} drivers, above MAX_DRIVERS={MAX_DRIVERS}")

    grid = np.full((len(grouped), MAX_DRIVERS), -1, dtype=np.int32)
    for i, rows in enumerate(grouped):
        grid[i, : len(rows)] = rows
    return grid, grid >= 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    argparse.ArgumentParser(description="Build the multi-task grid dataset.").parse_args()

    config.ensure_dirs()
    config.set_seed()

    seq_path = config.DATA_PROCESSED / "sequences.npz"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"{seq_path} not found. Run `python -m src.features.build_dataset` first."
        )

    print("Loading the Model 1/2 tensors ...")
    data = np.load(seq_path)
    meta = pd.read_parquet(config.DATA_PROCESSED / "meta.parquet")
    with open(config.DATA_PROCESSED / "feature_names.json") as fh:
        spec = json.load(fh)

    # Stack the three splits back into one array in meta order, so a single row
    # index addresses both X and meta.
    X = np.concatenate([data["X_train"], data["X_val"], data["X_test"]])
    y_pit = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    split = np.concatenate([
        np.full(len(data["X_train"]), "train"),
        np.full(len(data["X_val"]), "val"),
        np.full(len(data["X_test"]), "test"),
    ])
    if len(X) != len(meta):
        raise ValueError(f"X has {len(X)} rows but meta has {len(meta)}")
    # meta.parquet was written in train/val/test order, so the orders agree.
    if not (meta["split"].to_numpy() == split).all():
        raise ValueError("meta split order does not match the stacked tensor order")
    print(f"  X={X.shape}  meta={len(meta):,} rows")

    print("Recomputing auxiliary labels from the raw races ...")
    df = add_aux_labels(build_features(clean(load_raw()), spec["horizon"])[0])

    cols = ["y_compound", "y_compound_mask", "y_laptime", "y_laptime_mask",
            "y_outcome", "y_outcome_mask", "y_pit"]
    joined = meta[KEY].merge(df[KEY + cols], on=KEY, how="left", validate="one_to_one")
    if joined[cols].isna().any().any():
        missing = int(joined[cols].isna().any(axis=1).sum())
        raise ValueError(f"{missing} sample rows found no matching raw lap")
    # The rejoined y_pit must reproduce the one already in the tensors, which is
    # the check that the join preserved row order.
    if not (joined["y_pit"].to_numpy().astype(np.int8) == y_pit).all():
        raise ValueError("rejoined y_pit disagrees with sequences.npz -- row order is wrong")
    print("  join verified: y_pit reproduced exactly")

    y_compound = joined["y_compound"].to_numpy(np.int8)
    y_compound_mask = joined["y_compound_mask"].to_numpy(np.float32)
    y_laptime = joined["y_laptime"].to_numpy(np.float32)
    y_laptime_mask = joined["y_laptime_mask"].to_numpy(np.float32)
    y_outcome = joined["y_outcome"].to_numpy(np.int8)
    y_outcome_mask = joined["y_outcome_mask"].to_numpy(np.float32)

    # Standardise the regression target on the training split alone.
    is_train = split == "train"
    lt_fit = y_laptime[is_train & (y_laptime_mask > 0)]
    lt_mean, lt_std = float(lt_fit.mean()), float(lt_fit.std() or 1.0)
    y_laptime = ((y_laptime - lt_mean) / lt_std).astype(np.float32)
    # Clip the tail: a lap behind a safety car is 40s off the stint median and
    # would otherwise dominate the regression loss.
    y_laptime = np.clip(y_laptime, -5.0, 5.0)

    print("Grouping drivers into (race, lap) grids ...")
    grid, grid_mask = build_grid(meta)
    # Every row of a grid shares a race, so the split is constant within it.
    first = np.where(grid_mask, grid, 0)[:, 0]
    grid_split = split[first]
    per_grid = grid_mask.sum(1)
    print(f"  {len(grid):,} grids, {per_grid.mean():.1f} drivers each "
          f"(min {per_grid.min()}, max {per_grid.max()})")
    for name in ("train", "val", "test"):
        sel = grid_split == name
        rows_in = grid_mask[sel].sum()
        print(f"  {name:<5} {sel.sum():>6,} grids  {rows_in:>7,} driver-laps")
        covered = (split == name).sum()
        if rows_in != covered:
            raise ValueError(f"{name}: grids cover {rows_in} rows but the split holds {covered}")
    print("  coverage verified: every sample appears in exactly one grid")

    out = config.DATA_PROCESSED / "mtl_dataset.npz"
    np.savez_compressed(
        out,
        grid=grid, grid_mask=grid_mask, grid_split=grid_split,
        y_pit=y_pit,
        y_compound=y_compound, y_compound_mask=y_compound_mask,
        y_laptime=y_laptime, y_laptime_mask=y_laptime_mask,
        y_outcome=y_outcome, y_outcome_mask=y_outcome_mask,
        row_split=split,
    )
    with open(config.DATA_PROCESSED / "mtl_spec.json", "w") as fh:
        json.dump({"max_drivers": MAX_DRIVERS,
                   "n_compound_classes": len(COMPOUNDS),
                   "compounds": COMPOUNDS,
                   "n_outcome_classes": N_OUTCOME,
                   "laptime_mean": lt_mean,
                   "laptime_std": lt_std,
                   "seq_len": spec["seq_len"],
                   "horizon": spec["horizon"],
                   "n_features": len(spec["features"])}, fh, indent=2)

    print(f"\nSaved to {out}")
    print(f"  compound head supervised on {100 * y_compound_mask.mean():.1f}% of rows")
    print(f"  laptime  head supervised on {100 * y_laptime_mask.mean():.1f}% of rows")
    print(f"  outcome  head supervised on {100 * y_outcome_mask.mean():.1f}% of rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
