"""
Phase 7 -- export the 2024 season replay as a single JSON payload.

The interactive demo and the animated replay both read this file, so neither
re-runs a model: they replay predictions already verified against the published
metrics by ``src.evaluation.predict``.

Probabilities are rounded to three decimals, which is well below the resolution
any threshold decision turns on and keeps the payload small enough to embed
directly in a self-contained HTML page.

Usage
-----
    python -m src.evaluation.export_demo_data
"""

from __future__ import annotations

import json
import sys
import warnings

from src import config
from src.evaluation.compare import ORDER, SHORT
from src.evaluation.replay import actual_stop_laps, load_test_frame, season_summary

warnings.filterwarnings("ignore")

OUT = config.PROJECT_ROOT / "reports" / "demo" / "replay_data.json"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    test, horizon = load_test_frame()
    thresholds = test.attrs["thresholds"]

    races = {}
    for race_id, race_df in test.groupby("RaceId", sort=True):
        drivers = {}
        for drv, d in race_df.groupby("Driver", sort=True):
            d = d.sort_values("LapNumber")
            drivers[drv] = {
                "laps": [int(v) for v in d["LapNumber"]],
                "p": {name: [round(float(v), 3) for v in d[name]] for name in ORDER},
                "stops": [int(v) for v in actual_stop_laps(d, horizon)],
            }
        races[race_id] = {
            "name": str(race_df["EventName"].iloc[0]),
            "round": int(race_df["Round"].iloc[0]),
            "drivers": drivers,
        }

    summary = season_summary(test, horizon)
    payload = {
        "season": 2024,
        "horizon": horizon,
        "models": [{"key": k, "label": SHORT[k], "threshold": round(thresholds[k], 3)}
                   for k in ORDER],
        "season_summary": [
            {"model": r["model"], "anticipated": int(r["anticipated"]),
             "stops": int(r["stops_in_season"]), "pct": float(r["anticipated_pct"]),
             "median_lead": float(r["median_lead_laps"]),
             "false_alarms_per_driver_race": float(r["false_alarms_per_driver_race"])}
            for _, r in summary.iterrows()
        ],
        "races": races,
    }

    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    n_rows = sum(len(d["laps"]) for r in races.values() for d in r["drivers"].values())
    print(f"{OUT}")
    print(f"  {len(races)} races, {n_rows:,} driver-laps, {size_kb:,.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
