"""Equal-car GOAT counterfactual for the MIT Sloan 2027 abstract.

Two modes:

1. ``score_only`` (legacy) — Plackett--Luce Monte Carlo over exported peak-season
   driver Shapley skills + calibrated Gumbel noise. Artifact preserved for
   sensitivity comparison.

2. ``common_car_2025`` (default) — synthetic 2025-context common-car
   counterfactual: each champion keeps their peak-season final ``driver_state``
   (and career embedding); every 2025 race uses (a) the across-constructor mean
   embedding for that round as the shared car, (b) that race's context, and
   (c) a neutral common grid. Model utilities drive a 24-race PL season.

    python src/experiments/run_goat_equal_car.py \
      --mode common_car_2025 \
      --checkpoint output/orthogonal_shapley_model/orthogonal_shapley.pth \
      --meta output/orthogonal_shapley_model/orthogonal_shapley_meta.json \
      --baselines output/orthogonal_shapley_model/coalition_baselines.json \
      --max-year 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo/src/

from applications.goat_common_car import (  # noqa: E402
    NEUTRAL_GRID,
    assert_common_car_invariants,
    constructor_state_indices_for_race,
    counterfactual_race_utilities,
    mean_constructor_embedding,
    peak_career_indices,
    peak_driver_state_indices,
    race_ids_for_year,
    race_meta_for_id,
    simulate_season_from_utilities,
)
from baselines.orthogonal_shapley_skill import (  # noqa: E402
    export_orthogonal_shapley,
    get_orthogonal_shapley_db,
    load_orthogonal_shapley_model_and_graph,
)
from utils.naming import build_driver_name_map  # noqa: E402

F1_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)


# --------------------------------------------------------------------------- #
# Champion detection + peak-season rule
# --------------------------------------------------------------------------- #
def _season_end_standings(db) -> pd.DataFrame:
    standings = db.table_dict["standings"].df
    races = db.table_dict["races"].df[["raceId", "year", "round"]]
    df = standings.merge(races, on="raceId", how="inner")
    df = df.sort_values(["driverId", "year", "round"])
    season_end = df.groupby(["driverId", "year"], as_index=False).last()
    season_end["points"] = pd.to_numeric(season_end["points"], errors="coerce").fillna(0.0)
    season_end["position"] = pd.to_numeric(season_end["position"], errors="coerce")
    season_end["wins"] = pd.to_numeric(season_end["wins"], errors="coerce").fillna(0.0)
    total = season_end.groupby("year")["points"].transform("sum").replace(0.0, np.nan)
    season_end["share"] = (season_end["points"] / total).fillna(0.0)
    return season_end


def champion_set(
    db,
    *,
    min_champ_year: int = 1980,
    max_champ_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_end = _season_end_standings(db)
    names = build_driver_name_map(db.table_dict["drivers"].df)
    year_mask = season_end["year"] >= min_champ_year
    if max_champ_year is not None:
        year_mask &= season_end["year"] <= max_champ_year
    champs = season_end[year_mask & (season_end["position"] == 1.0)]
    rows = []
    for did in sorted(int(d) for d in champs["driverId"].unique()):
        title_years = sorted(int(y) for y in champs[champs["driverId"] == did]["year"])
        rows.append(
            {
                "driverId": did,
                "driver_name": names.get(did, f"driver_{did}"),
                "champion_years": title_years,
                "n_title_seasons": len(title_years),
            }
        )
    return pd.DataFrame(rows), season_end


def assign_peak_seasons(
    champions: pd.DataFrame,
    season_end: pd.DataFrame,
    skill: pd.DataFrame,
    *,
    peak_rule: str = "best_wdc_skill",
) -> pd.DataFrame:
    if peak_rule not in {"best_wdc_skill", "best_wdc_share"}:
        raise ValueError(f"unknown peak_rule: {peak_rule}")

    rows = []
    for _, champ in champions.iterrows():
        did = int(champ["driverId"])
        title_years = set(champ["champion_years"])
        title_rows = season_end[
            (season_end["driverId"] == did) & (season_end["year"].isin(title_years))
        ]
        if title_rows.empty:
            continue

        if peak_rule == "best_wdc_share":
            best_meta = title_rows.sort_values(
                ["share", "wins", "position"], ascending=[False, False, True]
            ).iloc[0]
            peak_season = int(best_meta["year"])
            peak_share = float(best_meta["share"])
            skill_row = skill[
                (skill["driverId"] == did) & (skill["season"] == peak_season)
            ]
        else:
            title_skill = skill[
                (skill["driverId"] == did) & (skill["season"].isin(title_years))
            ].dropna(subset=["skill_score"])
            if title_skill.empty:
                peak_season = int(
                    title_rows.sort_values(
                        ["share", "wins", "position"], ascending=[False, False, True]
                    ).iloc[0]["year"]
                )
                peak_share = float(
                    title_rows.loc[title_rows["year"] == peak_season, "share"].iloc[0]
                )
                skill_row = skill[
                    (skill["driverId"] == did) & (skill["season"] == peak_season)
                ]
            else:
                best_skill = title_skill.sort_values(
                    ["skill_score", "season"], ascending=[False, False]
                ).iloc[0]
                peak_season = int(best_skill["season"])
                peak_share = float(
                    title_rows.loc[title_rows["year"] == peak_season, "share"].iloc[0]
                )
                skill_row = best_skill.to_frame().T

        rows.append(
            {
                "driverId": did,
                "driver_name": champ["driver_name"],
                "champion_years": champ["champion_years"],
                "n_title_seasons": champ["n_title_seasons"],
                "peak_season": peak_season,
                "peak_share": peak_share,
                "skill_score": float(skill_row["skill_score"].iloc[0])
                if len(skill_row)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("driverId").reset_index(drop=True)


def calibrate_beta(export) -> float:
    race = export.race
    std = race.groupby(["driverId", "season"])["raw_skill"].std()
    beta = float(std.median())
    if not np.isfinite(beta) or beta <= 0.0:
        beta = float(std.std(ddof=0)) if std.notna().any() else 0.5
    return beta


# --------------------------------------------------------------------------- #
# Score-only Monte Carlo (legacy)
# --------------------------------------------------------------------------- #
def simulate_season(skills: np.ndarray, beta: float, dnf_rate: float,
                    n_races: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(skills)
    total = np.zeros(n)
    wins = np.zeros(n)
    points_table = np.asarray(F1_POINTS, dtype=float)
    for _ in range(n_races):
        u = skills + beta * rng.gumbel(0.0, 1.0, size=n)
        order = np.argsort(-u)
        dnf = rng.random(n) < dnf_rate
        pos = 0
        for i in order:
            if dnf[i]:
                continue
            if pos < len(points_table):
                total[i] += points_table[pos]
            if pos == 0:
                wins[i] += 1
            pos += 1
    return total, wins


def _champion_of(total: np.ndarray, wins: np.ndarray, rng: np.random.Generator) -> int:
    best = np.max(total)
    cand = np.flatnonzero(total == best)
    if cand.size == 1:
        return int(cand[0])
    cand = cand[wins[cand] == wins[cand].max()]
    if cand.size == 1:
        return int(cand[0])
    return int(rng.choice(cand))


def run_monte_carlo_skills(skills: np.ndarray, *, beta: float, dnf_rate: float,
                           n_races: int, n_sim: int, seed: int) -> dict:
    n = len(skills)
    rng = np.random.default_rng(seed)
    points_sum = np.zeros(n)
    points_sum2 = np.zeros(n)
    wins_sum = np.zeros(n)
    titles = np.zeros(n)

    for _ in range(n_sim):
        total, wins = simulate_season(skills, beta, dnf_rate, n_races, rng)
        points_sum += total
        points_sum2 += total ** 2
        wins_sum += wins
        titles[_champion_of(total, wins, rng)] += 1

    mean_points = points_sum / n_sim
    var_points = (points_sum2 / n_sim) - mean_points ** 2
    sd_points = np.sqrt(np.clip(var_points, 0.0, None))
    return {
        "p_title": titles / n_sim,
        "expected_points": mean_points,
        "points_lo95": mean_points - 1.96 * sd_points,
        "points_hi95": mean_points + 1.96 * sd_points,
        "expected_wins": wins_sum / n_sim,
    }


def run_monte_carlo_utilities(race_utilities: np.ndarray, *, beta: float, dnf_rate: float,
                              n_sim: int, seed: int) -> dict:
    n = race_utilities.shape[1]
    rng = np.random.default_rng(seed)
    points_sum = np.zeros(n)
    points_sum2 = np.zeros(n)
    wins_sum = np.zeros(n)
    titles = np.zeros(n)

    for _ in range(n_sim):
        total, wins = simulate_season_from_utilities(
            race_utilities, beta=beta, dnf_rate=dnf_rate, rng=rng
        )
        points_sum += total
        points_sum2 += total ** 2
        wins_sum += wins
        titles[_champion_of(total, wins, rng)] += 1

    mean_points = points_sum / n_sim
    var_points = (points_sum2 / n_sim) - mean_points ** 2
    sd_points = np.sqrt(np.clip(var_points, 0.0, None))
    return {
        "p_title": titles / n_sim,
        "expected_points": mean_points,
        "points_lo95": mean_points - 1.96 * sd_points,
        "points_hi95": mean_points + 1.96 * sd_points,
        "expected_wins": wins_sum / n_sim,
    }


def top5_kendall(order_a: np.ndarray, order_b: np.ndarray, k: int = 5) -> float:
    a = order_a[:k]
    b = order_b[:k]
    n = len(a)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom


def negative_controls_skills(skills: np.ndarray, *, beta: float, dnf_rate: float,
                             n_races: int, n_sim: int, seed: int) -> dict:
    out: dict = {}
    base = run_monte_carlo_skills(skills, beta=beta, dnf_rate=dnf_rate,
                                  n_races=n_races, n_sim=n_sim, seed=seed)
    shifted = run_monte_carlo_skills(skills + 2.5, beta=beta, dnf_rate=dnf_rate,
                                     n_races=n_races, n_sim=n_sim, seed=seed)
    delta = float(np.max(np.abs(shifted["p_title"] - base["p_title"])))
    out["shift_invariance"] = {
        "max_abs_delta_p_title": delta,
        "pass": bool(delta < 1e-9),
    }
    sym = run_monte_carlo_skills(
        np.array([1.0, 1.0]), beta=beta, dnf_rate=0.0,
        n_races=n_races, n_sim=n_sim, seed=seed,
    )
    se = np.sqrt(1.0 / n_sim)
    out["symmetry"] = {
        "p_title_pair": sym["p_title"].tolist(),
        "max_abs_delta": float(np.abs(sym["p_title"][0] - sym["p_title"][1])),
        "pass": bool(np.abs(sym["p_title"][0] - sym["p_title"][1]) < 3.0 * se),
    }
    return out


def negative_controls_utilities(race_utilities: np.ndarray, *, beta: float,
                                dnf_rate: float, n_sim: int, seed: int) -> dict:
    out: dict = {}
    base = run_monte_carlo_utilities(
        race_utilities, beta=beta, dnf_rate=dnf_rate, n_sim=n_sim, seed=seed
    )
    shifted = run_monte_carlo_utilities(
        race_utilities + 2.5, beta=beta, dnf_rate=dnf_rate, n_sim=n_sim, seed=seed
    )
    delta = float(np.max(np.abs(shifted["p_title"] - base["p_title"])))
    out["shift_invariance"] = {
        "max_abs_delta_p_title": delta,
        "pass": bool(delta < 1e-9),
    }
    n_races = race_utilities.shape[0]
    sym_u = np.ones((n_races, 2), dtype=float)
    sym = run_monte_carlo_utilities(
        sym_u, beta=beta, dnf_rate=0.0, n_sim=n_sim, seed=seed
    )
    se = np.sqrt(1.0 / n_sim)
    out["symmetry"] = {
        "p_title_pair": sym["p_title"].tolist(),
        "max_abs_delta": float(np.abs(sym["p_title"][0] - sym["p_title"][1])),
        "pass": bool(np.abs(sym["p_title"][0] - sym["p_title"][1]) < 3.0 * se),
    }
    return out


def sensitivity_skills(skills: np.ndarray, *, beta_grid, dnf_grid, n_races, n_sim, seed) -> dict:
    base_order = np.argsort(
        -run_monte_carlo_skills(skills, beta=float(np.median(beta_grid)),
                                dnf_rate=float(np.median(dnf_grid)),
                                n_races=n_races, n_sim=n_sim, seed=seed)["p_title"]
    )
    rows = []
    for b in beta_grid:
        for d in dnf_grid:
            res = run_monte_carlo_skills(skills, beta=float(b), dnf_rate=float(d),
                                        n_races=n_races, n_sim=n_sim, seed=seed)
            order = np.argsort(-res["p_title"])
            rows.append({
                "beta": float(b),
                "dnf_rate": float(d),
                "top5_kendall_vs_base": top5_kendall(base_order, order),
            })
    return {
        "base_beta": float(np.median(beta_grid)),
        "base_dnf": float(np.median(dnf_grid)),
        "grid": rows,
    }


def sensitivity_utilities(race_utilities: np.ndarray, *, beta_grid, dnf_grid,
                          n_sim, seed) -> dict:
    base_order = np.argsort(
        -run_monte_carlo_utilities(
            race_utilities, beta=float(np.median(beta_grid)),
            dnf_rate=float(np.median(dnf_grid)), n_sim=n_sim, seed=seed
        )["p_title"]
    )
    rows = []
    for b in beta_grid:
        for d in dnf_grid:
            res = run_monte_carlo_utilities(
                race_utilities, beta=float(b), dnf_rate=float(d),
                n_sim=n_sim, seed=seed,
            )
            order = np.argsort(-res["p_title"])
            rows.append({
                "beta": float(b),
                "dnf_rate": float(d),
                "top5_kendall_vs_base": top5_kendall(base_order, order),
            })
    return {
        "base_beta": float(np.median(beta_grid)),
        "base_dnf": float(np.median(dnf_grid)),
        "grid": rows,
    }


def render_figure(
    ranking: pd.DataFrame,
    out_path: Path,
    *,
    top_n: int = 6,
    peak_rule: str = "best_wdc_skill",
    title: str | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[figure] skipped: {exc}")
        return

    df = (
        ranking.nlargest(top_n, "expected_points")
        .iloc[::-1]
        .reset_index(drop=True)
    )
    y = np.arange(len(df))
    err = np.vstack([
        df["expected_points"] - df["points_lo95"],
        df["points_hi95"] - df["expected_points"],
    ])
    y_labels = [
        f"{name} ({int(season)})"
        for name, season in zip(df["driver_name"], df["peak_season"])
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
    ax.barh(
        y, df["expected_points"], xerr=err, capsize=3,
        color="#d71920", alpha=0.85, height=0.62,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Expected season points (equal car, 95% CI)", fontsize=10)
    peak_label = (
        "best WDC points share"
        if peak_rule == "best_wdc_share"
        else "best WDC-season skill"
    )
    ax.set_title(
        title or f"Equal-car GOAT: top 6 by expected points ({peak_label})",
        fontsize=12, loc="left", pad=10,
    )
    x_hi = float(df["points_hi95"].max())
    ax.set_xlim(0.0, x_hi * 1.08)
    ax.margins(x=0.02)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def _build_common_car_utilities(
    model,
    graph_data,
    x_dict,
    champions: pd.DataFrame,
    *,
    context_year: int,
    grid_value: float,
) -> tuple[np.ndarray, dict]:
    res = graph_data["results"]
    driver_ids = [int(d) for d in champions["driverId"].tolist()]
    peak_seasons = [int(s) for s in champions["peak_season"].tolist()]
    ds_map = peak_driver_state_indices(
        res, driver_ids=driver_ids, peak_seasons=peak_seasons
    )
    career_map = peak_career_indices(res, driver_ids=driver_ids)
    missing = [d for d in driver_ids if d not in ds_map]
    if missing:
        raise RuntimeError(f"missing peak driver_state for driverIds={missing}")

    race_ids = race_ids_for_year(res, context_year)
    if not race_ids:
        raise RuntimeError(f"no races found for context year {context_year}")

    ds_list = [ds_map[d] for d in driver_ids]
    career_list = [career_map[d] for d in driver_ids] if career_map else None

    util_rows = []
    mean_cars = []
    grids = []
    race_meta = []
    for rid in race_ids:
        c_idxs = constructor_state_indices_for_race(res, rid)
        mean_c = mean_constructor_embedding(x_dict, c_idxs)
        race_idx, round_num, year = race_meta_for_id(res, rid)
        u = counterfactual_race_utilities(
            model,
            x_dict,
            driver_state_indices=ds_list,
            career_indices=career_list,
            mean_constructor_emb=mean_c,
            race_idx=race_idx,
            round_num=round_num,
            grid_value=grid_value,
        )
        util_rows.append(u.detach().cpu().numpy())
        mean_cars.append(mean_c.detach().cpu())
        grids.append(torch.full((len(driver_ids),), grid_value))
        race_meta.append({
            "raceId": int(rid),
            "race_idx": int(race_idx),
            "round": int(round_num),
            "year": int(year),
            "n_constructors_in_mean": int(len(c_idxs)),
        })

    invariants = assert_common_car_invariants(
        race_ids=race_ids,
        per_race_constructor_means=mean_cars,
        grids=grids,
        grid_value=grid_value,
    )
    provenance = {
        "mode": "common_car_2025",
        "claim_level": "synthetic_2025_context_common_car_counterfactual",
        "context_year": context_year,
        "n_races": len(race_ids),
        "race_ids": race_ids,
        "race_meta": race_meta,
        "neutral_grid": grid_value,
        "driver_state_selection": "final_round_of_peak_WDC_season",
        "constructor_selection": "mean_constructor_state_embedding_per_2025_race",
        "peak_driver_state_idx": {str(d): ds_map[d] for d in driver_ids},
        "career_idx": {str(d): career_map.get(d) for d in driver_ids},
        "invariants": invariants,
    }
    return np.stack(util_rows, axis=0), provenance


def main() -> None:
    p = argparse.ArgumentParser(description="Equal-car GOAT counterfactual")
    p.add_argument(
        "--mode",
        choices=("common_car_2025", "score_only"),
        default="common_car_2025",
        help="common_car_2025 = model-forward 2025 contexts; score_only = legacy skill MC",
    )
    p.add_argument("--checkpoint", default="output/orthogonal_shapley_model/orthogonal_shapley.pth")
    p.add_argument("--meta", default="output/orthogonal_shapley_model/orthogonal_shapley_meta.json")
    p.add_argument("--baselines", default=None)
    p.add_argument("--max-year", type=int, default=2025)
    p.add_argument("--context-year", type=int, default=2025,
                   help="season whose race contexts define the synthetic calendar")
    p.add_argument("--min-champ-year", type=int, default=1980)
    p.add_argument("--n-races", type=int, default=24,
                   help="used by score_only; common_car_2025 uses all context-year races")
    p.add_argument("--n-sim", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dnf-rate", type=float, default=0.05)
    p.add_argument("--grid-value", type=float, default=NEUTRAL_GRID)
    p.add_argument(
        "--peak-rule",
        choices=("best_wdc_skill", "best_wdc_share"),
        default="best_wdc_skill",
    )
    p.add_argument("--output-dir", default="output/applications/goat_equal_car")
    p.add_argument("--gpu-id", type=int, default=None,
                   help="CUDA device id (default: config.DEFAULT_GPU_ID)")
    args = p.parse_args()

    db = get_orthogonal_shapley_db()
    print("-> detecting post-1980 world champions ...")
    champions, season_end = champion_set(
        db,
        min_champ_year=args.min_champ_year,
        max_champ_year=args.max_year,
    )
    print(f"   {len(champions)} champions")

    print("-> exporting skill scores from frozen model ...")
    export = export_orthogonal_shapley(
        db,
        checkpoint_path=args.checkpoint,
        meta_path=args.meta,
        baselines_path=args.baselines,
        max_year=args.max_year,
        gpu_id=args.gpu_id,
    )
    skill = export.season[["driverId", "season", "skill_score"]].dropna()
    beta = calibrate_beta(export)
    print(f"   calibrated beta = {beta:.4f}")

    print(f"-> assigning peak seasons ({args.peak_rule}) ...")
    champions = assign_peak_seasons(
        champions, season_end, skill, peak_rule=args.peak_rule,
    )
    missing = champions[champions["skill_score"].isna()]
    excluded = missing[["driver_name", "peak_season", "champion_years"]].to_dict("records")
    if len(missing):
        print(f"   WARNING: dropping {len(missing)} champions without peak skill")
    champions = champions.dropna(subset=["skill_score"]).reset_index(drop=True)

    mode_provenance: dict = {"mode": args.mode}
    if args.mode == "score_only":
        skills = champions["skill_score"].to_numpy(dtype=float)
        print(f"-> score_only Monte Carlo: {len(skills)} drivers, {args.n_races} races")
        res = run_monte_carlo_skills(
            skills, beta=beta, dnf_rate=args.dnf_rate,
            n_races=args.n_races, n_sim=args.n_sim, seed=args.seed,
        )
        controls = negative_controls_skills(
            skills, beta=beta, dnf_rate=args.dnf_rate,
            n_races=args.n_races, n_sim=args.n_sim, seed=args.seed,
        )
        beta_grid = [max(beta * f, 1e-3) for f in (0.5, 1.0, 2.0)]
        dnf_grid = [0.0, args.dnf_rate, 2 * args.dnf_rate]
        sens = sensitivity_skills(
            skills, beta_grid=beta_grid, dnf_grid=dnf_grid,
            n_races=args.n_races, n_sim=args.n_sim, seed=args.seed,
        )
        mode_provenance = {
            "mode": "score_only",
            "claim_level": "car_adjusted_performance_score_monte_carlo",
            "n_races": args.n_races,
            "note": "legacy: PL over exported peak season skills, not model-forward contexts",
        }
        fig_title = "Equal-car GOAT (score-only): top 6 by expected points"
        n_races_used = args.n_races
    else:
        print("-> loading model for common-car 2025 forward pass ...")
        model, graph_data, tf_dict, edge_index_dict, device, _baselines = (
            load_orthogonal_shapley_model_and_graph(
                db,
                checkpoint_path=args.checkpoint,
                meta_path=args.meta,
                baselines_path=args.baselines,
                gpu_id=args.gpu_id,
            )
        )
        model.eval()
        with torch.no_grad():
            x_dict = model.encode(tf_dict, edge_index_dict)
            race_utilities, mode_provenance = _build_common_car_utilities(
                model,
                graph_data,
                x_dict,
                champions,
                context_year=args.context_year,
                grid_value=args.grid_value,
            )
        n_races_used = int(race_utilities.shape[0])
        print(
            f"-> common_car_2025 Monte Carlo: {race_utilities.shape[1]} drivers, "
            f"{n_races_used} races from {args.context_year}"
        )
        res = run_monte_carlo_utilities(
            race_utilities, beta=beta, dnf_rate=args.dnf_rate,
            n_sim=args.n_sim, seed=args.seed,
        )
        controls = negative_controls_utilities(
            race_utilities, beta=beta, dnf_rate=args.dnf_rate,
            n_sim=args.n_sim, seed=args.seed,
        )
        beta_grid = [max(beta * f, 1e-3) for f in (0.5, 1.0, 2.0)]
        dnf_grid = [0.0, args.dnf_rate, 2 * args.dnf_rate]
        sens = sensitivity_utilities(
            race_utilities, beta_grid=beta_grid, dnf_grid=dnf_grid,
            n_sim=args.n_sim, seed=args.seed,
        )
        fig_title = (
            f"Equal-car GOAT (2025 contexts, common mean car): "
            f"top 6 by expected points"
        )

    champions = champions.assign(
        p_title=res["p_title"],
        expected_wins=res["expected_wins"],
        expected_points=res["expected_points"],
        points_lo95=res["points_lo95"],
        points_hi95=res["points_hi95"],
    )
    ranking = (
        champions.rename(columns={"skill_score": "skill"})
        .sort_values(["expected_points", "skill"], ascending=[False, False])
        .reset_index(drop=True)
    )

    db_max_year = int(db.table_dict["races"].df["year"].max())
    payload = {
        "config": vars(args),
        "provenance": {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "db_max_year": db_max_year,
            "max_champ_year": args.max_year,
            "claim_level": mode_provenance.get(
                "claim_level", "car_adjusted_performance"
            ),
            "repository": "https://github.com/AntonioNvs/ortogonal-representation",
            **mode_provenance,
        },
        "peak_rule": args.peak_rule,
        "calibrated_beta": beta,
        "n_champions": int(len(champions)),
        "n_races_used": n_races_used,
        "excluded_champions": excluded,
        "ranking": ranking[
            [
                "driver_name", "peak_season", "peak_share", "skill",
                "p_title", "expected_wins", "expected_points",
            ]
        ].to_dict("records"),
        "negative_controls": controls,
        "sensitivity": sens,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_name = (
        "goat_equal_car.json"
        if args.mode == "common_car_2025"
        else "goat_equal_car_score_only.json"
    )
    png_name = (
        "goat_equal_car.png"
        if args.mode == "common_car_2025"
        else "goat_equal_car_score_only.png"
    )
    with open(out_dir / json_name, "w") as f:
        json.dump(payload, f, indent=2, default=float)

    render_figure(
        ranking, out_dir / png_name, peak_rule=args.peak_rule, title=fig_title
    )

    print(f"\n=== Equal-car GOAT [{args.mode}] (top 6) ===")
    print(ranking.head(6).to_string(index=False))
    print(f"\ncontrols: {json.dumps(controls, indent=2, default=float)}")
    print(f"wrote {out_dir / json_name}")


if __name__ == "__main__":
    main()
