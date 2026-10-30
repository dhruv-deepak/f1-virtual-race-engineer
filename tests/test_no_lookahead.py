"""
Causality tests for the feature pipeline.

These exist because a real leak shipped. ``lap_time_delta_stint`` subtracted a
median taken over the *complete* stint, so a window ending at lap t carried a
statistic computed partly from laps the driver had not yet run -- including the
slow, degraded laps immediately before a pit stop. That made a pre-pit lap look
artificially quick against its own stint and flipped the feature's correlation
with the label from +0.036 to -0.072. Every reported score was inflated by it,
and nothing in the test suite or the metrics could have revealed it.

The check below is the general one: build features over a driver's full race,
then build them again over only the laps up to some lap t, and require every
feature value at lap t to be identical. A feature that reads the future fails,
whatever form the leak takes.

Run with:  python -m pytest tests/test_no_lookahead.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.build_dataset import build_features

# Features that are deliberately not causal and are excluded by name.
#   y_*            targets, which are future by definition
#   is_pit_out_lap known only once the driver has rejoined, but it describes
#                  lap t itself rather than any later lap
EXPECTED_NON_CAUSAL = {"y_pit"}


def _synthetic_race(n_laps=40, n_drivers=4, seed=0):
    """A small race with two stints per driver and a mid-race pit stop."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drivers):
        pit_lap = 18 + d          # each driver stops on a different lap
        for lap in range(1, n_laps + 1):
            stint = 1 if lap <= pit_lap else 2
            tyre_life = lap if stint == 1 else lap - pit_lap
            # Lap time degrades with tyre age, so the pre-pit laps are slow.
            lt = 90.0 + 0.25 * tyre_life + rng.normal(0, 0.15)
            rows.append({
                "RaceId": "2099_01", "Season": 2099, "Round": 1,
                "EventName": "Synthetic GP", "Driver": f"D{d}",
                "LapNumber": float(lap), "LapTime": lt, "Stint": float(stint),
                "TyreLife": float(tyre_life), "Compound": "MEDIUM",
                "FreshTyre": stint == 2, "IsPitInLap": lap == pit_lap,
                "IsPitOutLap": lap == pit_lap + 1, "Position": float(d + 1),
                "GridPosition": float(d + 1), "TotalLaps": n_laps,
                "Time": float(lap * 92 + d * 2),
                "Sector1Time": lt / 3, "Sector2Time": lt / 3, "Sector3Time": lt / 3,
                "SpeedI1": 300.0, "SpeedI2": 310.0, "SpeedFL": 290.0, "SpeedST": 320.0,
                "TrackStatus": "1", "AirTemp": 25.0, "TrackTemp": 40.0,
                "Humidity": 40.0, "Pressure": 1010.0, "WindSpeed": 3.0,
                "Rainfall": False, "FinalPosition": float(d + 1),
                "tel_speed_mean": 200.0, "tel_speed_max": 320.0,
                "tel_throttle_mean": 60.0, "tel_full_throttle_pct": 50.0,
                "tel_brake_pct": 20.0, "tel_rpm_mean": 10000.0, "tel_gear_mean": 5.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def race():
    return _synthetic_race()


@pytest.mark.parametrize("cut_lap", [14, 18, 19, 25, 33])
def test_features_at_lap_t_do_not_change_when_later_laps_are_removed(race, cut_lap):
    """
    The general causality check.

    Truncating the race after lap `cut_lap` must leave every feature value at
    lap `cut_lap` exactly as it was with the rest of the race present.
    """
    full, names = build_features(race.copy(), config.PIT_HORIZON)
    truncated, _ = build_features(
        race[race["LapNumber"] <= cut_lap].copy(), config.PIT_HORIZON)

    causal = [n for n in names if n not in EXPECTED_NON_CAUSAL]
    a = full[full["LapNumber"] == cut_lap].sort_values("Driver")[causal].to_numpy(float)
    b = truncated[truncated["LapNumber"] == cut_lap].sort_values("Driver")[causal].to_numpy(float)

    bad = [causal[j] for j in range(len(causal))
           if not np.allclose(a[:, j], b[:, j], rtol=1e-6, atol=1e-8, equal_nan=True)]
    assert not bad, (
        f"lap {cut_lap}: these features changed when future laps were removed, "
        f"so they read the future: {bad}"
    )


def test_stint_delta_is_expanding_not_whole_stint(race):
    """
    Pin the specific regression.

    A whole-stint median and an expanding median agree only on the final lap of
    a stint, so this would fail loudly if the original formulation returned.
    """
    out, _ = build_features(race.copy(), config.PIT_HORIZON)
    d = out[out["Driver"] == "D0"].sort_values("LapNumber")
    stint = d[d["Stint"] == 1]

    expanding = stint["LapTime"] - stint["LapTime"].expanding().median()
    np.testing.assert_allclose(
        stint["lap_time_delta_stint"].to_numpy(), expanding.to_numpy(),
        rtol=1e-6, atol=1e-8)

    whole = stint["LapTime"] - stint["LapTime"].median()
    assert not np.allclose(stint["lap_time_delta_stint"].to_numpy(),
                           whole.to_numpy(), rtol=1e-6), \
        "the feature matches a whole-stint median, which is the leak"


def test_label_never_depends_on_the_current_lap(race):
    """The pit label must come from laps strictly after t."""
    out, _ = build_features(race.copy(), config.PIT_HORIZON)
    for _, g in out.groupby(["RaceId", "Driver"], sort=False):
        g = g.sort_values("LapNumber")
        pit = g["is_pit_in_lap"].to_numpy()
        y = g["y_pit"].to_numpy()
        expect = np.array([1 if pit[t + 1:t + 1 + config.PIT_HORIZON].sum() > 0 else 0
                           for t in range(len(y))], dtype=np.int8)
        np.testing.assert_array_equal(y, expect)
