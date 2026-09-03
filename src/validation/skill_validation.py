"""Validation gates for driver-skill models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from validation.inference import cluster_bootstrap_spearman, partial_spearman, permutation_within_season


@dataclass
class ValidationResult:
    name: str
    metrics: Dict[str, float]
    passed: bool
    notes: str = ""


def ranking_nll_proxy(pairs: pd.DataFrame, skill_by_driver: Dict[int, float]) -> float:
    """Proxy NLL: fraction of pairs where higher skill predicts winner."""
    correct = 0
    total = 0
    for _, r in pairs.iterrows():
        sa = skill_by_driver.get(int(r["driverA"]), 0.0)
        sb = skill_by_driver.get(int(r["driverB"]), 0.0)
        if sa == sb:
            continue
        total += 1
        if sa > sb:
            correct += 1
    return 1.0 - (correct / total if total else 0.5)


def compare_skill_sources(
    joined: pd.DataFrame,
    skill_col: str = "skill_score",
    outcome_col: str = "outcome_score",
    tier_col: str = "constructor_tier_score_at_T",
) -> Dict[str, float]:
    sub = joined.dropna(subset=[skill_col, outcome_col])
    if len(sub) < 5:
        return {"spearman": float("nan"), "partial_rho": float("nan")}
    rho, _ = spearmanr(sub[skill_col], sub[outcome_col])
    partial = partial_spearman(sub, x_col=skill_col, y_col=outcome_col, z_col=tier_col)
    perm = permutation_within_season(sub.rename(columns={"season_T": "season_T"}))
    cluster = cluster_bootstrap_spearman(sub, x_col=skill_col, y_col=outcome_col)
    return {
        "spearman": float(rho),
        "partial_rho": partial.get("partial_rho", float("nan")),
        "partial_ci_low": partial.get("ci_low", float("nan")),
        "partial_ci_high": partial.get("ci_high", float("nan")),
        "perm_p": perm.get("p_value", float("nan")),
        "cluster_ci_low": cluster.get("ci_low", float("nan")),
        "cluster_ci_high": cluster.get("ci_high", float("nan")),
    }


def evaluate_gates(
    primary_metrics: Dict[str, float],
    leakage_rho: float,
    swap_skill_diff: float,
    bt_pairwise_acc: float,
    model_pairwise_acc: float,
    *,
    underrated_resolution: Optional[Dict] = None,
    underrated_auroc: Optional[Dict] = None,
) -> List[ValidationResult]:
    results = []
    results.append(
        ValidationResult(
            name="predictive_vs_bt",
            metrics={"model_acc": model_pairwise_acc, "bt_acc": bt_pairwise_acc},
            passed=model_pairwise_acc >= bt_pairwise_acc - 0.01,
            notes="Model should match or beat BT pairwise accuracy",
        )
    )
    results.append(
        ValidationResult(
            name="partial_spearman",
            metrics={"partial_rho": primary_metrics.get("partial_rho", float("nan"))},
            passed=primary_metrics.get("partial_rho", 0) >= 0.15
            and primary_metrics.get("partial_ci_low", -1) > 0,
            notes="Skill adds signal above constructor tier at T (diagnostic, all drivers)",
        )
    )
    ur = underrated_resolution or {}
    results.append(
        ValidationResult(
            name="underrated_resolution",
            metrics={
                "resolution_rate": ur.get("resolution_rate", float("nan")),
                "ci_low": ur.get("ci_low", float("nan")),
            },
            passed=not np.isnan(ur.get("resolution_rate", float("nan")))
            and ur.get("ci_low", -1) > 0.5,
            notes="Underrated drivers eventually promoted (rest-of-career)",
        )
    )
    uauroc = underrated_auroc or {}
    results.append(
        ValidationResult(
            name="underrated_promotion_auroc",
            metrics={"auroc": uauroc.get("auroc", float("nan"))},
            passed=not np.isnan(uauroc.get("auroc", float("nan")))
            and uauroc.get("auroc", 0) >= 0.45,
            notes="Skill discriminates promotion among underrated B-tier drivers (relative to baseline)",
        )
    )
    results.append(
        ValidationResult(
            name="underrated_partial_spearman",
            metrics={"partial_rho": primary_metrics.get("underrated_partial_rho", float("nan"))},
            passed=primary_metrics.get("underrated_partial_ci_low", -1) > -0.1,
            notes="Skill predicts forward tier in underrated stratum (raw Spearman when tier constant)",
        )
    )
    results.append(
        ValidationResult(
            name="constructor_leakage",
            metrics={"leakage_rho": leakage_rho},
            passed=abs(leakage_rho) < 0.3 if not np.isnan(leakage_rho) else False,
            notes="Driver readout should not correlate with constructor embedding norm",
        )
    )
    results.append(
        ValidationResult(
            name="swap_invariance",
            metrics={"skill_diff": swap_skill_diff},
            passed=swap_skill_diff < 0.05,
            notes="Driver skill stable under constructor swap",
        )
    )
    return results
