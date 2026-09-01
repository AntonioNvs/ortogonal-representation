"""Teammate-centered residual baseline: driver minus team mean per race."""

from __future__ import annotations

import pandas as pd

from data.race_panel import RacePanelConfig, build_race_panel
from relbench.base import Database
from skill.contract import InferenceMode
from skill.export import build_skill_export


def export_teammate_residual(db: Database, max_year: int = 2025):
    panel = build_race_panel(db, RacePanelConfig(max_year=max_year))
    ranked = panel[panel["in_race_ranking"]].copy()
    residuals = []
    for (race_id, constructor_id), grp in ranked.groupby(["raceId", "constructorId"]):
        team_mean = grp["race_skill"].mean()
        for _, row in grp.iterrows():
            residuals.append(
                {
                    **row.to_dict(),
                    "raw_skill": float(row["race_skill"] - team_mean),
                    "contrib_driver": float(row["race_skill"] - team_mean),
                    "contrib_constructor": float(team_mean),
                    "contrib_context": 0.0,
                    "contrib_residual": 0.0,
                    "as_of_round": int(row["round"]),
                    "support_bucket": "medium",
                }
            )
    race_df = pd.DataFrame(residuals)
    return build_skill_export(
        race_df,
        skill_source="teammate_residual",
        inference_mode=InferenceMode.FILTERED,
        max_year=max_year,
    )


def load_teammate_residual_skill(db: Database, max_year: int = 2025) -> pd.DataFrame:
    export = export_teammate_residual(db, max_year=max_year)
    return export.season[["driverId", "season", "skill_score"]]
