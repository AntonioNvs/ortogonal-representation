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
from validation.career_labels import driver_season_constructor, forward_tier_outcome
from validation.inference import (
    cluster_bootstrap_spearman,
    eligible_promotion_auroc,
    partial_spearman,
    permutation_within_season,
)
from validation.metrics import attach_race_positions, race_pl_nll_and_pairwise
from validation.skill_validation import compare_skill_sources, evaluate_gates
from validation.team_tiers import TIER_TO_SCORE, compute_constructor_season_points, compute_team_tiers


def join_career_from_export(export: SkillExport, db, team_tier: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    from experiments.evaluate_skill_model import join_career

    skill = export.season.copy()
    if "skill_score" not in skill.columns:
        skill["skill_score"] = skill.get("skill_0_10", skill.get("raw_skill"))
    return join_career(skill, db, team_tier, horizon=horizon)


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
    joined = join_career_from_export(export, db, team_tier)
    career = compare_skill_sources(joined)
    career["eligible_promotion_auroc"] = eligible_promotion_auroc(joined).get("auroc", float("nan"))
    career["cluster_bootstrap"] = cluster_bootstrap_spearman(joined)
    career["permutation"] = permutation_within_season(joined.rename(columns={"season_T": "season_T"}))
    report["career"] = career
    report["n_joined"] = len(joined)

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

    for source, rep in reports.items():
        if source in ("points_share", "constructor_tier"):
            continue
        locked = rep.get("locked_test", {})
        career = rep.get("career", {})
        partial_ok = career.get("partial_rho", 0) >= 0.15 and career.get("partial_ci_low", -1) > 0
        pl_ok = locked.get("pl_nll", float("inf")) <= bt_pl + 0.01
        acc_ok = locked.get("pairwise_acc", 0) >= bt_acc - 0.01
        gates.append(
            {
                "source": source,
                "partial_spearman_gate": partial_ok,
                "locked_pl_gate": pl_ok,
                "locked_pairwise_gate": acc_ok,
                "partial_rho": career.get("partial_rho"),
                "partial_ci_low": career.get("partial_ci_low"),
                "pl_nll": locked.get("pl_nll"),
                "pairwise_acc": locked.get("pairwise_acc"),
            }
        )
    return gates


def write_benchmark_report(reports: Dict[str, Dict], output_dir: str) -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)
    gates = evaluate_benchmark_gates(reports)
    payload = {"sources": reports, "gates": gates}
    with open(os.path.join(output_dir, "benchmark.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)

    lines = ["# Validation benchmark report", ""]
    for source, rep in reports.items():
        career = rep.get("career", {})
        locked = rep.get("locked_test", {})
        lines.append(f"## {source}")
        lines.append(f"- Partial ρ: {career.get('partial_rho', 'n/a')} (CI low: {career.get('partial_ci_low', 'n/a')})")
        lines.append(f"- Locked PL NLL: {locked.get('pl_nll', 'n/a')}")
        lines.append(f"- Locked pairwise acc: {locked.get('pairwise_acc', 'n/a')}")
        lines.append("")
    lines.append("## Gates")
    for g in gates:
        lines.append(f"- **{g['source']}**: partial={g['partial_spearman_gate']}, pl={g['locked_pl_gate']}, pairwise={g['locked_pairwise_gate']}")
    with open(os.path.join(output_dir, "benchmark.md"), "w") as f:
        f.write("\n".join(lines))
