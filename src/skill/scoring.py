"""Cumulative season skill helpers (model-agnostic)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_season_skill(race_df: pd.DataFrame) -> pd.DataFrame:
    """Causal as-of-round cumulative mean race_skill per (driver, season)."""
    rows = []
    for (driver_id, season), grp in race_df.groupby(["driverId", "year"]):
        grp = grp.sort_values("round")
        skills = []
        for _, r in grp.iterrows():
            skills.append(float(r["race_skill"]))
            rows.append(
                {
                    "driverId": int(driver_id),
                    "season": int(season),
                    "round": int(r["round"]),
                    "raceId": int(r["raceId"]),
                    "race_skill": float(r["race_skill"]),
                    "season_skill": float(np.mean(skills)),
                    "n_races": len(skills),
                    "constructorId": int(r["constructorId"]),
                    "driverRef": r.get("driverRef", ""),
                    "constructorRef": r.get("constructorRef", ""),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = (
        out.groupby(["season", "round"])["season_skill"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return out.sort_values(["season", "round", "rank"]).reset_index(drop=True)
