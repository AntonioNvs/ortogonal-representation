"""Shared career-validation metrics assembly."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from scipy.stats import spearmanr

from validation.inconsistency import build_inconsistency_report, mark_underrated
from validation.inference import (
    cluster_bootstrap_spearman,
    eligible_promotion_auroc,
    fisher_z_pooled,
    partial_spearman,
    permutation_within_season,
    stratum_partial_spearman,
)
from validation.skill_validation import compare_skill_sources


def parse_horizon_arg(horizon_str: str) -> Optional[int]:
    """Parse CLI horizon: 'inf' -> None (rest of career), else int."""
    if horizon_str.lower() in ("inf", "none", "null"):
        return None
    return int(horizon_str)


def compute_career_metrics(
    joined: pd.DataFrame,
    *,
    skill_source: str = "",
    horizon: Optional[int] = None,
    seed: int = 0,
    cohort_skill_col: str | None = None,
    min_year: int | None = None,
) -> dict:
    """Full career validation report for one joined table.

    ``cohort_skill_col`` (e.g. ``"cohort_skill_score"``) forces the underrated
    cohort to be defined on a model-free score column, so the flag is identical
    across models. When None, the cohort is defined on the model's own
    ``skill_score`` (endogenous — legacy behaviour).

    ``min_year`` filters rows to ``season_T >= min_year`` **before** flagging, so
    the within-season skill percentile and underrated flag are recomputed within
    the window (era stratification).
    """
    if min_year is not None and "season_T" in joined.columns:
        joined = joined[joined["season_T"] >= min_year].copy()
    marked = mark_underrated(joined, cohort_skill_col=cohort_skill_col)
    career = compare_skill_sources(marked)
    career["eligible_promotion_auroc"] = eligible_promotion_auroc(marked, seed=seed).get(
        "auroc", float("nan")
    )
    career["cluster_bootstrap"] = cluster_bootstrap_spearman(marked)
    career["permutation"] = permutation_within_season(marked)

    # Continuous car-quality control: residualize skill/outcome on the rolling
    # constructor points-share score (tighter than the 3-bin tier).
    if "constructor_score_at_T" in marked.columns and marked["constructor_score_at_T"].nunique() >= 2:
        _cont = partial_spearman(marked, z_col="constructor_score_at_T", seed=seed)
        career["partial_rho_continuous"] = _cont.get("partial_rho", float("nan"))
        career["partial_rho_continuous_ci_low"] = _cont.get("ci_low", float("nan"))
        career["partial_rho_continuous_ci_high"] = _cont.get("ci_high", float("nan"))
    else:
        career["partial_rho_continuous"] = float("nan")
        career["partial_rho_continuous_ci_low"] = float("nan")
        career["partial_rho_continuous_ci_high"] = float("nan")

    underrated_partial = stratum_partial_spearman(marked, seed=seed)
    career["underrated_partial_rho"] = underrated_partial.get("partial_rho", float("nan"))
    career["underrated_partial_ci_low"] = underrated_partial.get("ci_low", float("nan"))
    career["underrated_partial_ci_high"] = underrated_partial.get("ci_high", float("nan"))

    rising = marked[(marked["underrated_flag"]) & (marked["skill_slope_3yr"] > 0)].copy()
    if len(rising) >= 5:
        if rising["constructor_tier_score_at_T"].nunique() < 2:
            rising_result = cluster_bootstrap_spearman(rising, seed=seed)
            career["rising_underrated_partial_rho"] = rising_result.get("rho", float("nan"))
        else:
            rising_result = partial_spearman(rising, seed=seed)
            career["rising_underrated_partial_rho"] = rising_result.get("partial_rho", float("nan"))
    else:
        career["rising_underrated_partial_rho"] = float("nan")

    inconsistency = build_inconsistency_report(marked, seed=seed, cohort_skill_col=cohort_skill_col)
    marked_df = inconsistency.pop("marked_df")

    per_season = []
    for season, grp in marked.groupby("season_T"):
        if len(grp) < 5:
            continue
        rho, _ = spearmanr(grp["skill_score"], grp["outcome_score"])
        per_season.append({"season": int(season), "n": len(grp), "spearman": float(rho)})

    report = {
        "skill_source": skill_source,
        "horizon": "inf" if horizon is None else horizon,
        "n_rows": len(marked),
        "n_underrated": int(marked["underrated_flag"].sum()),
        "cluster_bootstrap": career["cluster_bootstrap"],
        "partial_spearman": partial_spearman(marked, seed=seed),
        "permutation": career["permutation"],
        "eligible_promotion_auroc": eligible_promotion_auroc(marked, seed=seed),
        "underrated_resolution": inconsistency["underrated_resolution"],
        "underrated_promotion_auroc": inconsistency["underrated_promotion_auroc"],
        "time_to_promotion": inconsistency["time_to_promotion"],
        "resolution_by_decade": inconsistency["resolution_by_decade"],
        "underrated_partial_spearman": underrated_partial,
        "career_summary": career,
        "fisher_z_pooled": fisher_z_pooled(
            pd.DataFrame(per_season) if per_season else pd.DataFrame()
        ),
    }
    return report, marked_df
