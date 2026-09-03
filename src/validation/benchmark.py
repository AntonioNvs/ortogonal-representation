"""Unified model-agnostic validation benchmark."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as cfg
from data.mobility import compute_mobility_report
from data.race_panel import RacePanelConfig, build_race_panel
from data.skill_dataset import SkillDatasetConfig
from explain.orthogonal_shapley_probes import run_orthogonal_shapley_probes
from explain.skill_gnn_probes import ProbeSampleConfig, infer_claim_level, run_xai_probes
from skill.calibration import distribution_diagnostics
from skill.contract import InferenceMode, SkillExport
from skill.decomposition import aggregate_season_shapley, bootstrap_shapley_ci, shapley_variance_shares
from validation.career_metrics import compute_career_metrics
from validation.inference import compare_resolution_rates
from validation.metrics import attach_race_positions, race_pl_nll_and_pairwise
from validation.team_tiers import TIER_TO_SCORE, compute_constructor_season_points, compute_team_tiers


def join_career_from_export(
    export: SkillExport,
    db,
    team_tier: pd.DataFrame,
    horizon: int | None = None,
    *,
    cohort_skill: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from experiments.evaluate_skill_model import join_career

    skill = export.season.copy()
    if "skill_score" not in skill.columns:
        skill["skill_score"] = skill.get("skill_0_10", skill.get("raw_skill"))
    return join_career(skill, db, team_tier, horizon=horizon, cohort_skill=cohort_skill)


def benchmark_source(
    export: SkillExport,
    db,
    team_tier: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    checkpoint_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    baselines_path: Optional[str] = None,
    xai_seed: int = 42,
    horizon: int | None = None,
    cohort_skill: pd.DataFrame | None = None,
    min_year: int | None = None,
    era_windows: bool = False,
) -> Dict:
    """Full benchmark report for one SkillExport."""
    source = export.source
    report: Dict = {
        "skill_source": source,
        "schema_version": export.metadata.schema_version,
        "inference_mode": export.metadata.inference_mode.value,
        "calibration": export.metadata.calibration,
    }

    # Contract / score behavior
    report["score_diagnostics"] = distribution_diagnostics(export.race["skill_0_10"].to_numpy())
    report["n_race_rows"] = len(export.race)
    report["n_season_rows"] = len(export.season)

    # Shapley decomposition (season aggregates)
    if all(c in export.race.columns for c in ("contrib_driver", "contrib_constructor", "contrib_context")):
        shapley = aggregate_season_shapley(export.race)
        shapley_ci = bootstrap_shapley_ci(export.race, seed=xai_seed)
        report["shapley_season_mean"] = {
            "driver": float(shapley["share_driver"].mean()) if not shapley.empty else float("nan"),
            "constructor": float(shapley["share_constructor"].mean()) if not shapley.empty else float("nan"),
            "context": float(shapley["share_context"].mean()) if not shapley.empty else float("nan"),
        }
        if not shapley_ci.empty:
            report["shapley_driver_share_ci"] = {
                "mean": float(shapley_ci["share_driver"].mean()),
                "lo": float(shapley_ci["share_driver_lo"].mean()),
                "hi": float(shapley_ci["share_driver_hi"].mean()),
            }

    # Locked test 2024-2025
    race_with_pos = attach_race_positions(export.race, panel)
    report["locked_test"] = race_pl_nll_and_pairwise(race_with_pos, test_years=tuple(cfg.TEST_YEARS))

    # Career validity
    joined = join_career_from_export(export, db, team_tier, horizon=horizon, cohort_skill=cohort_skill)
    cohort_skill_col = "cohort_skill_score" if cohort_skill is not None else None
    career_report, marked_joined = compute_career_metrics(
        joined,
        skill_source=source,
        horizon=horizon,
        seed=xai_seed,
        cohort_skill_col=cohort_skill_col,
        min_year=min_year,
    )
    report["cohort_skill_col"] = cohort_skill_col

    # Era stratification: modern (>=2010) primary, hybrid (>=1990) robustness.
    if era_windows and "season_T" in joined.columns:
        era = {}
        for label, ymin in (("modern_2010", 2010), ("hybrid_1990", 1990), ("full", None)):
            _rep, _ = compute_career_metrics(
                joined,
                skill_source=source,
                horizon=horizon,
                seed=xai_seed,
                cohort_skill_col=cohort_skill_col,
                min_year=ymin,
            )
            era[label] = {
                "n_rows": _rep.get("n_rows"),
                "n_underrated": _rep.get("n_underrated"),
                "partial_rho": _rep["career_summary"].get("partial_rho"),
                "partial_ci_low": _rep["career_summary"].get("partial_ci_low"),
                "partial_rho_continuous": _rep["career_summary"].get("partial_rho_continuous"),
                "partial_rho_continuous_ci_low": _rep["career_summary"].get("partial_rho_continuous_ci_low"),
                "underrated_resolution": _rep["underrated_resolution"].get("resolution_rate"),
                "underrated_auroc": _rep["underrated_promotion_auroc"].get("auroc"),
                "underrated_partial_rho": _rep["underrated_partial_spearman"].get("partial_rho"),
                "fisher_z_pooled": _rep["fisher_z_pooled"].get("rho_pooled"),
            }
        report["era_windows"] = era
    report["career"] = career_report["career_summary"]
    report["career"]["underrated_resolution"] = career_report["underrated_resolution"]
    report["career"]["underrated_promotion_auroc"] = career_report["underrated_promotion_auroc"]
    report["career"]["underrated_partial_rho"] = career_report["career_summary"].get(
        "underrated_partial_rho", float("nan")
    )
    report["career"]["underrated_partial_ci_low"] = career_report["career_summary"].get(
        "underrated_partial_ci_low", float("nan")
    )
    report["career"]["partial_rho_continuous"] = career_report["career_summary"].get(
        "partial_rho_continuous", float("nan")
    )
    report["career"]["partial_rho_continuous_ci_low"] = career_report["career_summary"].get(
        "partial_rho_continuous_ci_low", float("nan")
    )
    career = report["career"]
    report["career_full"] = {
        k: v for k, v in career_report.items() if k != "career_summary"
    }
    report["n_joined"] = len(joined)
    report["n_underrated"] = career_report.get("n_underrated", 0)

    # Survival (censored time-to-first-promotion): eligible primary, underrated secondary.
    from validation.survival import eligible_survival, survival_analysis

    try:
        report["survival"] = {
            "eligible": eligible_survival(marked_joined, seed=xai_seed),
            "underrated": survival_analysis(marked_joined, mask_col="underrated_flag", seed=xai_seed),
        }
    except Exception as exc:  # survival is diagnostic; never fatal
        report["survival_error"] = str(exc)

    # Mobility
    report["mobility"] = compute_mobility_report(db, SkillDatasetConfig(max_year=export.metadata.max_year)).to_dict()

    # Model XAI probes (optional)
    if source == "skill_gnn" and checkpoint_path:
        try:
            xai = run_xai_probes(
                db,
                checkpoint_path=checkpoint_path,
                meta_path=meta_path or "output/skill_model/skill_gnn_meta.json",
                config=ProbeSampleConfig(seed=xai_seed),
            )
            report["xai"] = xai
            report["claim_level"] = infer_claim_level(
                partial_rho=career.get("partial_rho", float("nan")),
                partial_ci_low=career.get("partial_ci_low", float("nan")),
                leakage_rho=xai["constructor_leakage_rho"],
            )
        except Exception as exc:
            report["xai_error"] = str(exc)
    elif source == "orthogonal_shapley":
        try:
            orth_ckpt = checkpoint_path or "output/orthogonal_shapley_model/orthogonal_shapley.pth"
            orth_meta = meta_path or "output/orthogonal_shapley_model/orthogonal_shapley_meta.json"
            xai = run_orthogonal_shapley_probes(
                db,
                checkpoint_path=orth_ckpt,
                meta_path=orth_meta,
                baselines_path=baselines_path,
                config=ProbeSampleConfig(seed=xai_seed),
            )
            report["xai"] = xai
            report["claim_level"] = infer_claim_level(
                partial_rho=career.get("partial_rho", float("nan")),
                partial_ci_low=career.get("partial_ci_low", float("nan")),
                leakage_rho=xai["constructor_leakage_rho"],
                leakage_pass=xai.get("gates", {}).get("constructor_leakage"),
            )
        except Exception as exc:
            report["xai_error"] = str(exc)

    return report


def evaluate_benchmark_gates(reports: Dict[str, Dict], *, bt_key: str = "bradley_terry") -> List[Dict]:
    """Apply predeclared gates; compare primary model vs best baseline PL."""
    gates = []
    bt = reports.get(bt_key, {})
    bt_pl = bt.get("locked_test", {}).get("pl_nll", float("inf"))
    bt_acc = bt.get("locked_test", {}).get("pairwise_acc", 0.0)
    bt_resolution = bt.get("career", {}).get("underrated_resolution", {}).get(
        "resolution_rate", float("nan")
    )

    for source, rep in reports.items():
        if source in ("points_share", "constructor_tier"):
            continue
        locked = rep.get("locked_test", {})
        career = rep.get("career", {})
        partial_ok = career.get("partial_rho", 0) >= 0.15 and career.get("partial_ci_low", -1) > 0
        pl_ok = locked.get("pl_nll", float("inf")) <= bt_pl + 0.01
        acc_ok = locked.get("pairwise_acc", 0) >= bt_acc - 0.01

        ur = career.get("underrated_resolution", {})
        resolution_rate = ur.get("resolution_rate", float("nan"))
        resolution_ci_low = ur.get("ci_low", float("nan"))
        resolution_ok = (
            not np.isnan(resolution_rate)
            and resolution_rate >= bt_resolution
            and resolution_ci_low > 0.5
        )

        uauroc = career.get("underrated_promotion_auroc", {})
        if isinstance(uauroc, dict):
            uauroc_val = uauroc.get("auroc", float("nan"))
            uauroc_ci_low = uauroc.get("ci_low", float("nan"))
        else:
            uauroc_val = float(uauroc)
            uauroc_ci_low = float("nan")
        bt_career = bt.get("career", {})
        bt_uauroc_obj = bt_career.get("underrated_promotion_auroc", {})
        bt_auroc_ref = (
            bt_uauroc_obj.get("auroc", 0)
            if isinstance(bt_uauroc_obj, dict)
            else float(bt_uauroc_obj)
        )
        uauroc_ok = not np.isnan(uauroc_val) and (
            source == bt_key
            or (uauroc_val >= bt_auroc_ref and uauroc_ci_low > 0.45)
        )

        underrated_partial_ok = (
            career.get("underrated_partial_rho", 0) >= bt_career.get("underrated_partial_rho", -1)
            and career.get("underrated_partial_ci_low", -1) > -0.1
        ) if source != bt_key else True

        gates.append(
            {
                "source": source,
                "partial_spearman_gate": partial_ok,
                "underrated_resolution_gate": resolution_ok,
                "underrated_auroc_gate": uauroc_ok,
                "underrated_partial_spearman_gate": underrated_partial_ok,
                "locked_pl_gate": pl_ok,
                "locked_pairwise_gate": acc_ok,
                "partial_rho": career.get("partial_rho"),
                "partial_ci_low": career.get("partial_ci_low"),
                "resolution_rate": resolution_rate,
                "resolution_ci_low": resolution_ci_low,
                "underrated_auroc": uauroc_val,
                "underrated_partial_rho": career.get("underrated_partial_rho"),
                "pl_nll": locked.get("pl_nll"),
                "pairwise_acc": locked.get("pairwise_acc"),
            }
        )
    return gates


def write_benchmark_report(reports: Dict[str, Dict], output_dir: str, extra: Dict | None = None) -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)
    gates = evaluate_benchmark_gates(reports)
    resolution_comparison = compare_resolution_rates(
        {s: r.get("career", {}) for s, r in reports.items()}
    )
    payload = {"sources": reports, "gates": gates, "resolution_comparison": resolution_comparison}
    if extra:
        payload.update(extra)
    with open(os.path.join(output_dir, "benchmark.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)

    lines = ["# Validation benchmark report", ""]
    for source, rep in reports.items():
        career = rep.get("career", {})
        locked = rep.get("locked_test", {})
        ur = career.get("underrated_resolution", {})
        lines.append(f"## {source}")
        lines.append(f"- Partial ρ: {career.get('partial_rho', 'n/a')} (CI low: {career.get('partial_ci_low', 'n/a')})")
        lines.append(
            f"- Underrated resolution: {ur.get('resolution_rate', 'n/a')} "
            f"(CI low: {ur.get('ci_low', 'n/a')}, n={ur.get('n_underrated', 'n/a')})"
        )
        uauroc = career.get("underrated_promotion_auroc", {})
        if isinstance(uauroc, dict):
            lines.append(f"- Underrated promotion AUROC: {uauroc.get('auroc', 'n/a')}")
        lines.append(f"- Locked PL NLL: {locked.get('pl_nll', 'n/a')}")
        lines.append(f"- Locked pairwise acc: {locked.get('pairwise_acc', 'n/a')}")
        lines.append("")
    lines.append("## Gates")
    for g in gates:
        lines.append(
            f"- **{g['source']}**: partial={g['partial_spearman_gate']}, "
            f"resolution={g['underrated_resolution_gate']}, "
            f"underrated_auroc={g['underrated_auroc_gate']}, "
            f"pl={g['locked_pl_gate']}, pairwise={g['locked_pairwise_gate']}"
        )
    with open(os.path.join(output_dir, "benchmark.md"), "w") as f:
        f.write("\n".join(lines))
