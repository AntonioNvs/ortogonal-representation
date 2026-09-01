"""Model-agnostic skill export contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

RACE_COLUMNS = [
    "driverId",
    "season",
    "round",
    "raceId",
    "constructorId",
    "lineage_id",
    "driver_name",
    "constructor_name",
    "raw_skill",
    "skill_0_10",
    "skill_lo",
    "skill_hi",
    "contrib_driver",
    "contrib_constructor",
    "contrib_context",
    "contrib_residual",
    "skill_source",
    "inference_mode",
    "as_of_round",
]

SEASON_COLUMNS = [
    "driverId",
    "season",
    "skill_score",
    "skill_0_10",
    "skill_lo",
    "skill_hi",
    "skill_source",
    "inference_mode",
    "as_of_round",
    "n_obs",
    "support_bucket",
]


class InferenceMode(str, Enum):
    FILTERED = "filtered"  # causal: only data through round R
    SMOOTHED = "smoothed"  # descriptive posterior / full-interval only


@dataclass
class SkillExportMetadata:
    skill_source: str
    inference_mode: InferenceMode
    dnf_policy: str
    calibration: Dict[str, float]
    train_years: List[int]
    max_year: int
    as_of_round: Optional[int] = None
    walk_forward: bool = True
    schema_version: str = "1.0"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skill_source": self.skill_source,
            "inference_mode": self.inference_mode.value,
            "dnf_policy": self.dnf_policy,
            "calibration": self.calibration,
            "train_years": self.train_years,
            "max_year": self.max_year,
            "as_of_round": self.as_of_round,
            "walk_forward": self.walk_forward,
            "schema_version": self.schema_version,
            **self.extra,
        }


@dataclass
class SkillExport:
    source: str
    metadata: SkillExportMetadata
    race: pd.DataFrame
    season: pd.DataFrame

    def validate(self) -> None:
        missing_race = set(RACE_COLUMNS) - set(self.race.columns)
        if missing_race:
            raise ValueError(f"race export missing columns: {sorted(missing_race)}")
        missing_season = set(SEASON_COLUMNS) - set(self.season.columns)
        if missing_season:
            raise ValueError(f"season export missing columns: {sorted(missing_season)}")
        if self.metadata.inference_mode == InferenceMode.FILTERED:
            if self.race["as_of_round"].isna().any():
                raise ValueError("filtered mode requires as_of_round on all race rows")


def empty_race_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RACE_COLUMNS)


def empty_season_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SEASON_COLUMNS)
