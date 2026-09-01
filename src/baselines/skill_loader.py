"""Central registry loading SkillExport for all sources."""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

import config as cfg
from baselines.bradley_terry_skill import export_bradley_terry
from baselines.bayesian_ssm import export_bayesian_ssm
from baselines.skill_gnn_skill import export_skill_gnn
from baselines.teammate_residual import export_teammate_residual
from data.enriched_dataset import EnrichedF1Dataset
from skill.contract import InferenceMode, SkillExport
from validation.baselines import export_constructor_tier, export_points_share


def get_validation_db():
    from baselines.skill_gnn_skill import get_skill_gnn_db

    return get_skill_gnn_db()


def load_skill_export(
    source: str,
    db=None,
    *,
    max_year: int = 2025,
    inference_mode: InferenceMode = InferenceMode.FILTERED,
    output_dir: Optional[str] = None,
    checkpoint_path: str = "output/skill_model/skill_gnn.pth",
    meta_path: str = "output/skill_model/skill_gnn_meta.json",
    force_recompute: bool = False,
) -> SkillExport:
    """Load or compute a validated SkillExport for the given source."""
    if db is None:
        db = get_validation_db()

    cache_dir = output_dir or os.path.join("output/skill_exports", source)
    meta_path_disk = os.path.join(cache_dir, "metadata.json")
    race_path = os.path.join(cache_dir, "race_skill.parquet")
    season_path = os.path.join(cache_dir, "season_skill.csv")

    if (
        not force_recompute
        and os.path.isfile(meta_path_disk)
        and os.path.isfile(race_path)
        and os.path.isfile(season_path)
    ):
        with open(meta_path_disk) as f:
            meta = json.load(f)
        race = pd.read_parquet(race_path)
        season = pd.read_csv(season_path)
        from skill.contract import SkillExportMetadata

        metadata = SkillExportMetadata(
            skill_source=meta["skill_source"],
            inference_mode=InferenceMode(meta["inference_mode"]),
            dnf_policy=meta["dnf_policy"],
            calibration=meta["calibration"],
            train_years=meta["train_years"],
            max_year=meta["max_year"],
            as_of_round=meta.get("as_of_round"),
            walk_forward=meta.get("walk_forward", True),
            schema_version=meta.get("schema_version", "1.0"),
            extra={k: v for k, v in meta.items() if k not in {
                "skill_source", "inference_mode", "dnf_policy", "calibration",
                "train_years", "max_year", "as_of_round", "walk_forward", "schema_version",
            }},
        )
        export = SkillExport(source=source, metadata=metadata, race=race, season=season)
        export.validate()
        return export

    if source == "bradley_terry":
        export = export_bradley_terry(db, max_year=max_year, inference_mode=inference_mode)
    elif source == "bayesian_ssm" or source == "bayesian_comparator":
        export = export_bayesian_ssm(
            db,
            start_year=2014,
            end_year=min(2021, max_year),
            inference_mode=inference_mode,
            output_dir=cache_dir,
        )
    elif source == "skill_gnn":
        export = export_skill_gnn(
            db,
            checkpoint_path=checkpoint_path,
            meta_path=meta_path,
            max_year=max_year,
            inference_mode=inference_mode,
        )
    elif source == "teammate_residual":
        export = export_teammate_residual(db, max_year=max_year)
    elif source == "points_share":
        export = export_points_share(db, max_year=max_year)
    elif source == "constructor_tier":
        export = export_constructor_tier(db, max_year=max_year)
    else:
        raise ValueError(f"unknown skill source: {source}")

    os.makedirs(cache_dir, exist_ok=True)
    export.race.to_parquet(race_path, index=False)
    export.season.to_csv(season_path, index=False)
    with open(meta_path_disk, "w") as f:
        json.dump(export.metadata.to_dict(), f, indent=2)
    return export


def season_scores_for_career(export: SkillExport) -> pd.DataFrame:
    """Career join uses skill_score column (raw) plus calibrated score."""
    df = export.season.copy()
    if "skill_score" not in df.columns and "skill_0_10" in df.columns:
        df["skill_score"] = df["skill_0_10"]
    return df.rename(columns={"season": "season"})
