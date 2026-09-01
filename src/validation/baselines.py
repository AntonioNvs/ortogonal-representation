"""Skill-scorer baselines depending only on the raw DB."""

from __future__ import annotations

import pandas as pd

from data.race_panel import RacePanelConfig, build_race_panel
from skill.contract import InferenceMode
from skill.export import build_skill_export
from validation.team_tiers import TIER_TO_SCORE


def load_points_share(db) -> pd.DataFrame:
    standings = db.table_dict["standings"].df
    races = db.table_dict["races"].df[["raceId", "year", "round"]]
    df = standings.merge(races, on="raceId", how="inner")
    df = df.sort_values(["driverId", "year", "round"])
    season_end = df.groupby(["driverId", "year"], as_index=False).last()
    season_end = season_end.rename(columns={"year": "season"})
    season_end["points"] = pd.to_numeric(season_end["points"], errors="coerce").fillna(0.0)
    season_max = season_end.groupby("season")["points"].transform("max")
    season_end["skill_score"] = season_end["points"] / season_max.replace(0.0, float("nan"))
    season_end["skill_score"] = season_end["skill_score"].fillna(0.0)
    return season_end[["driverId", "season", "skill_score"]].sort_values(["driverId", "season"]).reset_index(drop=True)


def load_constructor_tier(db, team_tier: pd.DataFrame) -> pd.DataFrame:
    from validation.career_labels import driver_season_constructor

    ds = driver_season_constructor(db)
    tier_lookup = team_tier.set_index(["constructorId", "season"])["tier"].to_dict()
    ds["tier"] = [
        tier_lookup.get((int(cid), int(s))) for cid, s in zip(ds["constructorId"], ds["season"])
    ]
    ds = ds.dropna(subset=["tier"])
    ds["skill_score"] = ds["tier"].map(TIER_TO_SCORE).astype(float)
    return ds[["driverId", "season", "skill_score"]].sort_values(["driverId", "season"]).reset_index(drop=True)


def _simple_race_export(panel: pd.DataFrame, raw_col: str, source: str) -> "SkillExport":
    race_df = panel[panel["in_race_ranking"]].copy()
    race_df["raw_skill"] = race_df[raw_col].astype(float)
    race_df["contrib_driver"] = race_df["raw_skill"]
    race_df["contrib_constructor"] = 0.0
    race_df["contrib_context"] = 0.0
    race_df["contrib_residual"] = 0.0
    race_df["as_of_round"] = race_df["round"]
    race_df["support_bucket"] = "medium"
    return build_skill_export(
        race_df,
        skill_source=source,
        inference_mode=InferenceMode.FILTERED,
    )


def export_points_share(db, max_year: int = 2025):
    panel = build_race_panel(db, RacePanelConfig(max_year=max_year))
    panel["points_proxy"] = panel.get("race_skill", 0.0)
    return _simple_race_export(panel, "points_proxy", "points_share")


def export_constructor_tier(db, max_year: int = 2025):
    from validation.team_lineage import lineage_id_by_constructor
    from validation.team_tiers import compute_constructor_season_points, compute_team_tiers

    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
    tiers = compute_team_tiers(compute_constructor_season_points(db), lineage=lineage)
    panel = build_race_panel(db, RacePanelConfig(max_year=max_year))
    tier_map = tiers.set_index(["constructorId", "season"])["tier"].to_dict()
    panel["tier_score"] = [
        TIER_TO_SCORE.get(tier_map.get((int(c), int(s)), "B"), 1.0)
        for c, s in zip(panel["constructorId"], panel["season"])
    ]
    return _simple_race_export(panel, "tier_score", "constructor_tier")
