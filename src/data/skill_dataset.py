"""Canonical race-level dataset for driver-skill ranking.

One row per ``(driverId, constructorId, raceId)`` appearance, merging qualifying,
results, race metadata, and status. Supports multiple DNF policies and normalized
rank targets for ranking likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from data.temporal_graph import deduplicate_results
from relbench.base import Database


class DnfPolicy(str, Enum):
    """Which rows enter the race-ranking likelihood."""

    CLASSIFIED = "classified"  # position not null (primary)
    ALL_ENTRIES = "all_entries"  # use positionOrder for everyone who started
    FINISHED = "finished"  # statusId == 1 (crossing the line)


# StatusId 1 = Finished (Ergast convention).
STATUS_FINISHED = 1


@dataclass(frozen=True)
class SkillDatasetConfig:
    min_year: int = 1950
    max_year: int = 2025  # headline eval excludes incomplete 2026
    dnf_policy: DnfPolicy = DnfPolicy.CLASSIFIED


def _normalize_rank(
    position: np.ndarray | pd.Series | float,
    session_size: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Map rank 1..n to skill score in [0,1] where 1 = pole/winner.

    Vectorized: rows with session_size <= 1 or NaN position become NaN.
    """
    pos = np.atleast_1d(np.asarray(position, dtype=np.float64))
    size = np.atleast_1d(np.asarray(session_size, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - (pos - 1.0) / (size - 1.0)
    invalid = (size <= 1.0) | np.isnan(pos) | np.isnan(size)
    out[invalid] = np.nan
    if np.ndim(position) == 0 and np.ndim(session_size) == 0:
        return np.float64(out[0])
    return out


def build_skill_dataset(db: Database, config: Optional[SkillDatasetConfig] = None) -> pd.DataFrame:
    """Build the canonical skill table from a RelBench Database.

    Returns a DataFrame with one row per race entry and columns:
        driverId, constructorId, raceId, year, round, circuitId,
        qualifying_position, qualifying_skill, grid, race_position,
        race_position_order, race_skill, classified, finished,
        n_qualifiers, n_race_entries, statusId, driverRef, constructorRef
    """
    cfg = config or SkillDatasetConfig()

    drivers = db.table_dict["drivers"].df
    constructors = db.table_dict["constructors"].df
    races = db.table_dict["races"].df
    qualifying = db.table_dict["qualifying"].df
    results = db.table_dict["results"].df

    race_meta = races.set_index("raceId")[["year", "round", "circuitId", "name"]]

    qual = qualifying.merge(race_meta, left_on="raceId", right_index=True, how="inner")
    qual = qual[(qual["year"] >= cfg.min_year) & (qual["year"] <= cfg.max_year)]

    res = results.merge(race_meta, left_on="raceId", right_index=True, how="inner")
    res = res[(res["year"] >= cfg.min_year) & (res["year"] <= cfg.max_year)]

    # Dedup results on (driverId, raceId) — shared with temporal_graph.
    res = deduplicate_results(res)

    n_qual = qual.groupby("raceId")["qualifyId"].transform("size").astype(float)
    qual["qualifying_skill"] = _normalize_rank(qual["position"].astype(float), n_qual)

    n_race = res.groupby("raceId")["resultId"].transform("size").astype(float)
    res["race_skill_raw"] = _normalize_rank(res["position"].astype(float), n_race)
    res["race_skill_order"] = _normalize_rank(res["positionOrder"].astype(float), n_race)

    merged = res.merge(
        qual[
            [
                "raceId",
                "driverId",
                "constructorId",
                "position",
                "qualifying_skill",
            ]
        ].rename(columns={"position": "qualifying_position", "constructorId": "qual_constructorId"}),
        on=["raceId", "driverId"],
        how="left",
        suffixes=("", "_qual"),
    )
    # Prefer results constructor (actual race team); fall back to qualifying team.
    merged["constructorId"] = merged["constructorId"].fillna(merged["qual_constructorId"])

    merged = merged.merge(
        drivers[["driverId", "driverRef", "forename", "surname"]],
        on="driverId",
        how="left",
    )
    merged = merged.merge(
        constructors[["constructorId", "constructorRef", "name"]].rename(
            columns={"name": "constructorName"}
        ),
        on="constructorId",
        how="left",
    )

    merged["classified"] = merged["position"].notna()
    merged["finished"] = merged["statusId"] == STATUS_FINISHED

    if cfg.dnf_policy == DnfPolicy.CLASSIFIED:
        merged["race_skill"] = merged["race_skill_raw"]
        merged["in_race_ranking"] = merged["classified"]
    elif cfg.dnf_policy == DnfPolicy.ALL_ENTRIES:
        merged["race_skill"] = merged["race_skill_order"]
        merged["in_race_ranking"] = True
    else:
        merged["race_skill"] = np.where(
            merged["finished"], merged["race_skill_raw"], np.nan
        )
        merged["in_race_ranking"] = merged["finished"]

    qual_counts = qual.groupby("raceId")["qualifyId"].count()
    race_counts = res.groupby("raceId")["resultId"].count()
    merged["n_qualifiers"] = merged["raceId"].map(qual_counts).fillna(0).astype(int)
    merged["n_race_entries"] = merged["raceId"].map(race_counts).fillna(0).astype(int)

    cols = [
        "driverId",
        "driverRef",
        "forename",
        "surname",
        "constructorId",
        "constructorRef",
        "constructorName",
        "raceId",
        "year",
        "round",
        "circuitId",
        "qualifying_position",
        "qualifying_skill",
        "grid",
        "position",
        "positionOrder",
        "race_skill",
        "classified",
        "finished",
        "in_race_ranking",
        "statusId",
        "n_qualifiers",
        "n_race_entries",
    ]
    out = merged[cols].copy()
    out = out.rename(columns={"position": "race_position", "positionOrder": "race_position_order"})
    out = out.sort_values(["year", "round", "race_position_order"]).reset_index(drop=True)
    return out


def build_qualifying_only(db: Database, config: Optional[SkillDatasetConfig] = None) -> pd.DataFrame:
    """Qualifying rows for drivers who set a time (no race result required)."""
    cfg = config or SkillDatasetConfig()
    races = db.table_dict["races"].df
    qualifying = db.table_dict["qualifying"].df
    race_meta = races.set_index("raceId")[["year", "round", "circuitId"]]

    qual = qualifying.merge(race_meta, left_on="raceId", right_index=True)
    qual = qual[(qual["year"] >= cfg.min_year) & (qual["year"] <= cfg.max_year)]
    n_qual = qual.groupby("raceId")["qualifyId"].transform("size").astype(float)
    qual["qualifying_skill"] = _normalize_rank(qual["position"].astype(float), n_qual.to_numpy())
    return qual.sort_values(["year", "round", "position"]).reset_index(drop=True)


def filter_by_years(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    allowed = set(years)
    return df[df["year"].isin(allowed)].copy()


def assert_skill_dataset_invariants(df: pd.DataFrame) -> None:
    """Raise AssertionError if basic invariants fail."""
    dup = df.duplicated(subset=["driverId", "raceId"], keep=False)
    assert not dup.any(), f"duplicate (driverId, raceId): {int(dup.sum())} rows"

    assert df["year"].notna().all()
    assert (df["round"] >= 1).all()

    # Chronological round within season
    for year, grp in df.groupby("year"):
        rounds = sorted(grp["round"].unique())
        assert rounds == list(range(min(rounds), max(rounds) + 1)) or len(rounds) >= 1

    # Qualifying skill in [0,1] when present
    q = df["qualifying_skill"].dropna()
    if len(q):
        assert q.min() >= -1e-6 and q.max() <= 1.0 + 1e-6

    ranked = df.loc[df["in_race_ranking"], "race_skill"].dropna()
    if len(ranked):
        assert ranked.min() >= -1e-6 and ranked.max() <= 1.0 + 1e-6
