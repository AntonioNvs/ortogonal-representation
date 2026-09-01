"""Cumulative season skill helpers (model-agnostic)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_season_skill(race_df: pd.DataFrame) -> pd.DataFrame:
    """Causal as-of-round cumulative mean skill per (driver, season)."""
    df = race_df.copy()
    season_col = "season" if "season" in df.columns else "year"
    skill_col = "skill_0_10" if "skill_0_10" in df.columns else "raw_skill" if "raw_skill" in df.columns else "race_skill"

    rows = []
    for (driver_id, season), grp in df.groupby(["driverId", season_col]):
        grp = grp.sort_values("round")
        skills = []
        for _, r in grp.iterrows():
            skills.append(float(r[skill_col]))
            rows.append(
                {
                    "driverId": int(driver_id),
                    "season": int(season),
                    "round": int(r["round"]),
                    "raceId": int(r["raceId"]),
                    "race_skill": float(r[skill_col]),
                    "season_skill": float(np.mean(skills)),
                    "n_races": len(skills),
                    "constructorId": int(r["constructorId"]),
                    "driverRef": r.get("driverRef", r.get("driver_name", "")),
                    "constructorRef": r.get("constructorRef", r.get("constructor_name", "")),
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
