"""Canonical (driver, constructor, race) panel for validation and baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data.skill_dataset import SkillDatasetConfig, build_skill_dataset
from relbench.base import Database
from utils.naming import build_constructor_name_map, build_driver_name_map
from validation.team_lineage import build_lineage_map, lineage_id_by_constructor


@dataclass(frozen=True)
class RacePanelConfig:
    min_year: int = 1950
    max_year: int = 2025
    dnf_policy: str = "classified"


def _demean_within_race(series: pd.Series, race_ids: pd.Series) -> pd.Series:
    df = pd.DataFrame({"v": series, "raceId": race_ids})
    means = df.groupby("raceId")["v"].transform("mean")
    return series - means


def _best_qualifying_lap(qual_row: pd.Series) -> float:
    """Fastest available session time; NaN if no lap-time columns."""
    for col in ("q3", "q2", "q1"):
        if col not in qual_row.index:
            continue
        val = pd.to_numeric(qual_row[col], errors="coerce")
        if pd.notna(val):
            return float(val)
    return float("nan")


def _standardize_qualifying_lap(df: pd.DataFrame, db: Database) -> pd.Series:
    """Within-GP z-score of qualifying performance (lower = better).

    Uses fastest lap time when ``q1``/``q2``/``q3`` exist in the DB; otherwise
    falls back to ``qualifying.position`` (available in RelBench/F1DB exports).
    """
    qualifying = db.table_dict["qualifying"].df
    races = db.table_dict["races"].df[["raceId", "year", "round"]]
    q = qualifying.merge(races, on="raceId", how="inner")
    q = q.sort_values(["driverId", "raceId", "position"])
    fastest = q.groupby(["driverId", "raceId"], as_index=False).first()

    has_lap_times = any(c in fastest.columns for c in ("q1", "q2", "q3"))
    if has_lap_times:
        fastest["q_metric"] = fastest.apply(_best_qualifying_lap, axis=1)
        impute_factor = 1.07
    else:
        fastest["q_metric"] = pd.to_numeric(fastest["position"], errors="coerce")
        impute_factor = 1.0

    race_bounds = fastest.groupby("raceId")["q_metric"].agg(["min", "max"])
    imputed = []
    for _, row in fastest.iterrows():
        metric = row["q_metric"]
        rid = row["raceId"]
        bounds = race_bounds.loc[rid]
        if pd.isna(metric):
            if has_lap_times and pd.notna(bounds["min"]):
                metric = impute_factor * float(bounds["min"])
            elif pd.notna(bounds["max"]):
                metric = float(bounds["max"]) + 1.0
        imputed.append(metric)
    fastest["q_metric_adj"] = imputed

    z_scores = []
    for _, grp in fastest.groupby("raceId"):
        vals = grp["q_metric_adj"].astype(float)
        mu = vals.mean()
        sd = vals.std(ddof=0)
        if sd <= 1e-9:
            z = pd.Series(0.0, index=grp.index)
        else:
            z = (vals - mu) / sd
        z_scores.append(z)
    fastest["qualifying_z"] = pd.concat(z_scores).sort_index()
    return fastest.set_index(["driverId", "raceId"])["qualifying_z"]


def build_race_panel(db: Database, config: Optional[RacePanelConfig] = None) -> pd.DataFrame:
    """One row per classified race entry with display names and lineage."""
    cfg = config or RacePanelConfig()
    skill_cfg = SkillDatasetConfig(
        min_year=cfg.min_year,
        max_year=cfg.max_year,
    )
    df = build_skill_dataset(db, skill_cfg)
    drivers = db.table_dict["drivers"].df
    constructors = db.table_dict["constructors"].df
    races = db.table_dict["races"].df[["raceId", "name"]]

    driver_names = build_driver_name_map(drivers)
    constructor_names = build_constructor_name_map(constructors)
    lineage_map = build_lineage_map(constructors)
    cid_to_lineage = lineage_id_by_constructor(constructors)

    df["driver_name"] = df["driverId"].map(driver_names)
    df["constructor_name"] = df["constructorId"].map(constructor_names)
    df["lineage_id"] = df["constructorId"].map(cid_to_lineage)
    df = df.merge(races.rename(columns={"name": "race_name"}), on="raceId", how="left")
    df = df.rename(columns={"year": "season"})
    df["grid_demeaned"] = _demean_within_race(df["grid"].astype(float), df["raceId"])

    try:
        qz = _standardize_qualifying_lap(df, db)
        df = df.set_index(["driverId", "raceId"])
        df["qualifying_z"] = qz.reindex(df.index)
        df = df.reset_index()
    except Exception:
        df["qualifying_z"] = np.nan

    df["inference_mode"] = "filtered"
    df["as_of_round"] = df["round"]
    return df.sort_values(["season", "round", "race_position_order"]).reset_index(drop=True)


def filter_panel_through_round(panel: pd.DataFrame, season: int, max_round: int) -> pd.DataFrame:
    """Causal as-of-round subset for season."""
    sub = panel[(panel["season"] == season) & (panel["round"] <= max_round)].copy()
    sub["as_of_round"] = max_round
    sub["inference_mode"] = "filtered"
    return sub
