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
) -> dict:
    """Full career validation report for one joined table."""
    marked = mark_underrated(joined)
    career = compare_skill_sources(marked)
    career["eligible_promotion_auroc"] = eligible_promotion_auroc(marked, seed=seed).get(
        "auroc", float("nan")
    )
    career["cluster_bootstrap"] = cluster_bootstrap_spearman(marked)
    career["permutation"] = permutation_within_season(marked)

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

    inconsistency = build_inconsistency_report(marked, seed=seed)
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
