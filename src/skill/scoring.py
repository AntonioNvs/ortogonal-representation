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


def peak_season_skill(race_df: pd.DataFrame) -> pd.DataFrame:
    """Season-mean skill per (driver, season): peak = average skill across the season."""
    if race_df.empty:
        return pd.DataFrame(
            columns=[
                "driverId",
                "season",
                "peak_skill",
                "n_races",
                "driver_name",
                "constructor_name",
            ]
        )

    season_col = "season" if "season" in race_df.columns else "year"
    skill_col = (
        "skill_0_10"
        if "skill_0_10" in race_df.columns
        else "raw_skill"
        if "raw_skill" in race_df.columns
        else "race_skill"
    )

    peaks = (
        race_df.groupby(["driverId", season_col], as_index=False)
        .agg(peak_skill=(skill_col, "mean"), n_races=(skill_col, "size"))
        .rename(columns={season_col: "season"})
    )

    if "driver_name" in race_df.columns:
        name_agg: dict[str, tuple[str, str]] = {"driver_name": ("driver_name", "first")}
        if "constructor_name" in race_df.columns:
            name_agg["constructor_name"] = ("constructor_name", "first")
        names = (
            race_df.groupby(["driverId", season_col], as_index=False)
            .agg(**name_agg)
            .rename(columns={season_col: "season"})
        )
        peaks = peaks.merge(names, on=["driverId", "season"], how="left")

    return peaks.sort_values(
        ["peak_skill", "season", "driverId"], ascending=[False, False, True]
    ).reset_index(drop=True)
