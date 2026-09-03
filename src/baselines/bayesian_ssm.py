"""Bayesian state-space driver/constructor model (Lindner et al. 2026).

Faithful structure: joint qualifying (Normal) + race (Plackett-Luce), GP-level
random-walk abilities, demeaned grid, sum-to-zero constraints. Inference via Stan NUTS.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from data.race_panel import RacePanelConfig, build_race_panel
from relbench.base import Database
from skill.contract import InferenceMode
from skill.export import build_skill_export

STAN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bayesian_ssm.stan")


@dataclass
class StanFitConfig:
    chains: int = 4
    iter_warmup: int = 2000
    iter_sampling: int = 2000
    seed: int = 42
    adapt_delta: float = 0.95
    max_treedepth: int = 12


def _prepare_stan_data(panel: pd.DataFrame, start_year: int, end_year: int) -> dict:
    sub = panel[(panel["season"] >= start_year) & (panel["season"] <= end_year)].copy()
    sub = sub[sub["in_race_ranking"]].sort_values(["season", "round", "race_position_order"])
    gps = sub[["season", "round", "raceId"]].drop_duplicates().sort_values(["season", "round"])
    gps["gp_idx"] = np.arange(1, len(gps) + 1)
    sub = sub.merge(gps[["raceId", "gp_idx"]], on="raceId", how="left")

    drivers = sorted(sub["driverId"].unique())
    constructors = sorted(sub["constructorId"].unique())
    d_map = {d: i + 1 for i, d in enumerate(drivers)}
    c_map = {c: i + 1 for i, c in enumerate(constructors)}

    sub["d_idx"] = sub["driverId"].map(d_map)
    sub["c_idx"] = sub["constructorId"].map(c_map)

    qual = sub.dropna(subset=["qualifying_z"])
    race = sub.copy()

    # Build race ranking groups
    race_groups = []
    for rid, grp in race.groupby("raceId"):
        grp = grp.sort_values("race_position_order")
        race_groups.append(len(grp))

    return {
        "N_qual": len(qual),
        "N_race_obs": len(race),
        "N_races": len(race_groups),
        "D": len(drivers),
        "K": len(constructors),
        "T": int(gps["gp_idx"].max()),
        "qual_driver": qual["d_idx"].astype(int).tolist(),
        "qual_constructor": qual["c_idx"].astype(int).tolist(),
        "qual_gp": qual["gp_idx"].astype(int).tolist(),
        "y_qual": qual["qualifying_z"].astype(float).tolist(),
        "race_driver": race["d_idx"].astype(int).tolist(),
        "race_constructor": race["c_idx"].astype(int).tolist(),
        "race_gp": race["gp_idx"].astype(int).tolist(),
        "grid": race["grid_demeaned"].astype(float).tolist(),
        "finish_pos": race["race_position_order"].astype(int).tolist(),
        "race_size": race_groups,
        "driver_id_map": {int(k): int(v) for k, v in d_map.items()},
        "constructor_id_map": {int(k): int(v) for k, v in c_map.items()},
        "panel": sub,
        "gps": gps,
    }


def _check_mcmc_diagnostics(fit) -> dict:
    try:
        import arviz as az

        idata = az.from_cmdstanpy(fit)
        summary = az.summary(idata)
        rhat_max = float(summary["r_hat"].max()) if "r_hat" in summary.columns else float("nan")
        ess_min = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else float("nan")
        divergences = int(fit.num_divergences) if hasattr(fit, "num_divergences") else 0
        return {
            "rhat_max": rhat_max,
            "ess_bulk_min": ess_min,
            "divergences": divergences,
            "passed": divergences == 0 and rhat_max < 1.01 and ess_min >= 400,
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def _fit_stan(data: dict, stan_config: StanFitConfig, stan_file: str):
    try:
        from cmdstanpy import CmdStanModel
    except ImportError as exc:
        raise ImportError(
            "cmdstanpy is required for bayesian_ssm. Install with: pip install cmdstanpy arviz"
        ) from exc

    model = CmdStanModel(stan_file=stan_file)
    fit = model.sample(
        data={k: v for k, v in data.items() if k not in ("panel", "gps", "driver_id_map", "constructor_id_map")},
        chains=stan_config.chains,
        iter_warmup=stan_config.iter_warmup,
        iter_sampling=stan_config.iter_sampling,
        seed=stan_config.seed,
        adapt_delta=stan_config.adapt_delta,
        max_treedepth=stan_config.max_treedepth,
        show_progress=True,
    )
    return fit


def _posterior_race_export(
    prep: dict,
    fit,
    *,
    inference_mode: InferenceMode,
) -> pd.DataFrame:
    """Map posterior mean abilities to race-level export rows."""
    a_mean = fit.stan_variable("a").mean(axis=0)  # (D, T)
    c_mean = fit.stan_variable("c").mean(axis=0)  # (K, T)
    beta_grid = float(fit.stan_variable("beta_grid").mean())

    panel = prep["panel"]
    inv_d = {v: k for k, v in prep["driver_id_map"].items()}
    inv_c = {v: k for k, v in prep["constructor_id_map"].items()}

    rows = []
    for _, r in panel.iterrows():
        di = int(r["d_idx"]) - 1
        ci = int(r["c_idx"]) - 1
        ti = int(r["gp_idx"]) - 1
        driver_eff = float(-a_mean[di, ti])  # lower latent = better in paper; flip to higher-is-better
        cons_eff = float(-c_mean[ci, ti])
        grid_eff = float(-beta_grid * r["grid_demeaned"])
        context = grid_eff
        raw = driver_eff
        rows.append(
            {
                "driverId": int(r["driverId"]),
                "season": int(r["season"]),
                "round": int(r["round"]),
                "raceId": int(r["raceId"]),
                "constructorId": int(r["constructorId"]),
                "lineage_id": r.get("lineage_id", ""),
                "driver_name": r.get("driver_name", ""),
                "constructor_name": r.get("constructor_name", ""),
                "raw_skill": raw,
                "contrib_driver": driver_eff,
                "contrib_constructor": cons_eff,
                "contrib_context": context,
                "contrib_residual": 0.0,
                "as_of_round": int(r["round"]),
                "inference_mode": inference_mode.value,
                "support_bucket": "medium",
            }
        )
    return pd.DataFrame(rows)


def export_bayesian_ssm(
    db: Database,
    *,
    start_year: int = 2014,
    end_year: int = 2025,
    inference_mode: InferenceMode = InferenceMode.SMOOTHED,
    output_dir: Optional[str] = None,
    stan_config: Optional[StanFitConfig] = None,
    smoke_test: bool = False,
) -> "SkillExport":
    """Fit Lindner-style SSM and return SkillExport."""
    stan_config = stan_config or StanFitConfig()
    if smoke_test:
        stan_config = StanFitConfig(chains=1, iter_warmup=50, iter_sampling=50, seed=stan_config.seed)

    panel = build_race_panel(db, RacePanelConfig(min_year=start_year, max_year=end_year))
    prep = _prepare_stan_data(panel, start_year, end_year)
    if prep["N_qual"] < 1:
        raise ValueError(
            f"No qualifying observations between {start_year} and {end_year}. "
            "Ensure qualifying.position (or q1/q2/q3 lap times) is present in the database."
        )
    if prep["N_races"] < 1 or prep["N_race_obs"] < 2:
        raise ValueError(
            f"Insufficient race observations between {start_year} and {end_year} "
            f"(N_races={prep['N_races']}, N_race_obs={prep['N_race_obs']})."
        )
    stan_path = os.path.abspath(STAN_MODEL_PATH)
    if not os.path.isfile(stan_path):
        raise FileNotFoundError(f"Stan model not found: {stan_path}")

    fit = _fit_stan(prep, stan_config, stan_path)
    diagnostics = _check_mcmc_diagnostics(fit)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fit.save_csvfiles(dir=output_dir)
        with open(os.path.join(output_dir, "mcmc_diagnostics.json"), "w") as f:
            json.dump(diagnostics, f, indent=2)

    race_df = _posterior_race_export(prep, fit, inference_mode=inference_mode)
    return build_skill_export(
        race_df,
        skill_source="bayesian_ssm",
        inference_mode=inference_mode,
        max_year=end_year,
        walk_forward=False,
        extra_meta={"mcmc_diagnostics": diagnostics, "start_year": start_year, "end_year": end_year},
    )


def load_bayesian_comparator_skill(db: Database, max_year: int = 2025) -> pd.DataFrame:
    """Backward-compatible alias — loads cached bayesian_ssm export if present."""
    from baselines.skill_loader import load_skill_export

    export = load_skill_export("bayesian_ssm", db, max_year=max_year)
    return export.season[["driverId", "season", "skill_score"]].rename(columns={"skill_score": "skill_score"})
