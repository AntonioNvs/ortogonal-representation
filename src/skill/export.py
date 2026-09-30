"""Normalize model outputs into the common SkillExport contract."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from skill.calibration import CalibrationParams, calibrate_interval, calibrate_to_0_10, fit_calibration
from skill.contract import (
    InferenceMode,
    SEASON_COLUMNS,
    SkillExport,
    SkillExportMetadata,
    empty_race_frame,
    empty_season_frame,
)


def apply_calibration_to_race_df(
    race_df: pd.DataFrame,
    params: CalibrationParams,
    *,
    skill_source: str,
    inference_mode: InferenceMode,
    raw_col: str = "raw_skill",
) -> pd.DataFrame:
    """Add skill_0_10 and propagate lo/hi if present."""
    out = race_df.copy()
    out["skill_0_10"] = calibrate_to_0_10(out[raw_col].to_numpy(), params)
    if "skill_lo" in out.columns and "skill_hi" in out.columns:
        lo_hi = [
            calibrate_interval(float(r.skill_lo), float(r.skill_hi), params)
            if pd.notna(r.skill_lo) and pd.notna(r.skill_hi)
            else (np.nan, np.nan)
            for r in out.itertuples(index=False)
        ]
        out["skill_lo"] = [x[0] for x in lo_hi]
        out["skill_hi"] = [x[1] for x in lo_hi]
    else:
        out["skill_lo"] = np.nan
        out["skill_hi"] = np.nan
    out["skill_source"] = skill_source
    out["inference_mode"] = inference_mode.value
    return out


def race_to_season_summary(
    race_df: pd.DataFrame,
    *,
    skill_source: str,
    inference_mode: InferenceMode,
) -> pd.DataFrame:
    """Aggregate race-level export to season table for career validation."""
    if race_df.empty:
        return empty_season_frame()

    rows = []
    for (driver_id, season), grp in race_df.groupby(["driverId", "season"]):
        rows.append(
            {
                "driverId": int(driver_id),
                "season": int(season),
                "skill_score": float(grp["raw_skill"].mean()),
                "skill_0_10": float(grp["skill_0_10"].mean()),
                "skill_lo": float(grp["skill_lo"].mean()) if grp["skill_lo"].notna().any() else float("nan"),
                "skill_hi": float(grp["skill_hi"].mean()) if grp["skill_hi"].notna().any() else float("nan"),
                "skill_source": skill_source,
                "inference_mode": inference_mode.value,
                "as_of_round": int(grp["as_of_round"].max()) if "as_of_round" in grp.columns else None,
                "n_obs": int(len(grp)),
                "support_bucket": grp["support_bucket"].iloc[0]
                if "support_bucket" in grp.columns
                else None,
            }
        )
    out = pd.DataFrame(rows)
    for col in SEASON_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[SEASON_COLUMNS].sort_values(["driverId", "season"]).reset_index(drop=True)


def build_skill_export(
    race_df: pd.DataFrame,
    *,
    skill_source: str,
    inference_mode: InferenceMode,
    dnf_policy: str = "classified",
    train_years: Optional[list[int]] = None,
    max_year: int = 2025,
    calibration: Optional[CalibrationParams] = None,
    walk_forward: bool = True,
    extra_meta: Optional[dict] = None,
) -> SkillExport:
    """Fit calibration on train-years raw_skill and return validated export."""
    train_years = train_years if train_years is not None else list(cfg.TRAIN_YEARS)
    if calibration is None:
        train_mask = race_df["season"].isin(train_years) if "season" in race_df.columns else race_df["year"].isin(train_years)
        calibration = fit_calibration(race_df.loc[train_mask, "raw_skill"])

    race = apply_calibration_to_race_df(
        race_df,
        calibration,
        skill_source=skill_source,
        inference_mode=inference_mode,
    )
    season = race_to_season_summary(race, skill_source=skill_source, inference_mode=inference_mode)
    meta = SkillExportMetadata(
        skill_source=skill_source,
        inference_mode=inference_mode,
        dnf_policy=dnf_policy,
        calibration=calibration.to_dict(),
        train_years=train_years,
        max_year=max_year,
        walk_forward=walk_forward,
        extra=extra_meta or {},
    )
    export = SkillExport(source=skill_source, metadata=meta, race=race, season=season)
    export.validate()
    return export
