"""Equal-car GOAT counterfactual for the MIT Sloan 2027 abstract.

Ranks post-1980 Formula 1 world champions by title probability in a common-car
counterfactual season, using the frozen orthogonal-Shapley skill model.

Design (locked, see docs/plans/2026-09-15-sloan-abstract-final-experiments.md §5):

1. **Champion set** is derived from the data, not hardcoded: a driver is a
   "world champion" if they finished the driver standings with position 1 in any
   season >= ``--min-champ-year`` (default 1980).
2. **Peak season** is chosen by an *external* rule, never by the model: the
   season with the largest points share of that season's total (era-adjusted by
   construction), tie-broken by wins then position. This keeps the GOAT ranking
   independent of the skill model that produces the numbers.
3. **Skill** is the frozen model's season skill at that externally chosen peak
   (mean within-race-centred driver Shapley contribution over the season).
4. **Equal-car race** is a Plackett--Luce draw over the champions' peak skills:
   ``utility_i = skill_i + beta * Gumbel(0,1)``. ``beta`` is calibrated to the
   historical within-(driver,season) race-level residual std of the model, so
   race-to-race variance is data-driven rather than assumed. DNF is an
   independent per-race dropout at rate ``--dnf-rate``. Points use the modern F1
   table (25-18-15-12-10-8-6-4-2-1) over the top finishers.

Negative controls (always run and written to the JSON):
  * shift-invariance — adding a constant to every skill must leave title
    probabilities unchanged (softmax is shift-invariant);
  * symmetry — two drivers with equal skill must have equal title probability;
  * sensitivity — title ranking is re-evaluated over a ``beta`` and ``dnf`` grid
    and its stability is reported (top-5 Kendall tau).

In the additive readout the driver Shapley channel ``u_d(d) - u_d(baseline)`` is
car-free by construction, so "everyone in the same car" is already what the
driver-skill score measures; the Monte Carlo only converts it into season-level
title/win/points probabilities.

Run on the A100:

    python src/experiments/run_goat_equal_car.py \
      --checkpoint output/orthogonal_shapley_model/orthogonal_shapley.pth \
      --meta output/orthogonal_shapley_model/orthogonal_shapley_meta.json \
      --baselines output/orthogonal_shapley_model/coalition_baselines.json \
      --max-year 2025
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo/src/

from baselines.orthogonal_shapley_skill import (  # noqa: E402
    export_orthogonal_shapley,
    get_orthogonal_shapley_db,
)
from utils.naming import build_driver_name_map  # noqa: E402

F1_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)


# --------------------------------------------------------------------------- #
# Champion detection + external peak-season rule
# --------------------------------------------------------------------------- #
def champion_peak_seasons(db, *, min_champ_year: int = 1980) -> pd.DataFrame:
    """Post-1980 champions with an externally chosen representative season.

    Returns one row per champion with ``driverId``, ``driver_name``,
    ``champion_years``, ``peak_season`` and ``peak_share``.
    """
    standings = db.table_dict["standings"].df
    races = db.table_dict["races"].df[["raceId", "year", "round"]]
    names = build_driver_name_map(db.table_dict["drivers"].df)

    df = standings.merge(races, on="raceId", how="inner")
    df = df.sort_values(["driverId", "year", "round"])
    season_end = df.groupby(["driverId", "year"], as_index=False).last()
    season_end["points"] = pd.to_numeric(season_end["points"], errors="coerce").fillna(0.0)
    season_end["position"] = pd.to_numeric(season_end["position"], errors="coerce")
    season_end["wins"] = pd.to_numeric(season_end["wins"], errors="coerce").fillna(0.0)

    champs = season_end[
        (season_end["year"] >= min_champ_year) & (season_end["position"] == 1.0)
    ]
    champ_ids = sorted(int(d) for d in champs["driverId"].unique())

    total = season_end.groupby("year")["points"].transform("sum").replace(0.0, np.nan)
    season_end["share"] = (season_end["points"] / total).fillna(0.0)

    rows = []
    for did in champ_ids:
        car = season_end[season_end["driverId"] == did]
        if car.empty:
            continue
        best = car.sort_values(
            ["share", "wins", "position"], ascending=[False, False, True]
        ).iloc[0]
        rows.append(
            {
                "driverId": did,
                "driver_name": names.get(did, f"driver_{did}"),
                "champion_years": sorted(
                    int(y) for y in champs[champs["driverId"] == did]["year"]
                ),
                "peak_season": int(best["year"]),
                "peak_share": float(best["share"]),
                "n_seasons": int(len(car)),
            }
        )
    return pd.DataFrame(rows).sort_values("driverId").reset_index(drop=True)


def load_season_skill(db, *, checkpoint, meta, baselines, max_year) -> pd.DataFrame:
    export = export_orthogonal_shapley(
        db,
        checkpoint_path=checkpoint,
        meta_path=meta,
        baselines_path=baselines,
        max_year=max_year,
    )
    return export


def calibrate_beta(export) -> float:
    """Data-driven race-to-race noise scale.

    ``beta`` = median within-(driver, season) standard deviation of the
    race-level ``raw_skill``. This is the model's own estimate of how much a
    driver's per-race skill varies around their seasonal mean, so the simulated
    season inherits realistic race-to-race variance.
    """
    race = export.race
    std = race.groupby(["driverId", "season"])["raw_skill"].std()
    beta = float(std.median())
    if not np.isfinite(beta) or beta <= 0.0:
        beta = float(std.std(ddof=0)) if std.notna().any() else 0.5
    return beta


# --------------------------------------------------------------------------- #
# Equal-car season Monte Carlo
# --------------------------------------------------------------------------- #
def simulate_season(skills: np.ndarray, beta: float, dnf_rate: float,
                    n_races: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one common-car season; return (total_points, wins) per driver."""
    n = len(skills)
    total = np.zeros(n)
    wins = np.zeros(n)
    points_table = np.asarray(F1_POINTS, dtype=float)
    for _ in range(n_races):
        # Plackett--Luce: utility = skill + Gumbel(0, beta)
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
    # lexicographic: max points, then max wins, with a *random* tie-break so that
    # identical drivers do not get a deterministic index bias.
    best = np.max(total)
    cand = np.flatnonzero(total == best)
    if cand.size == 1:
        return int(cand[0])
    cand = cand[wins[cand] == wins[cand].max()]
    if cand.size == 1:
        return int(cand[0])
    return int(rng.choice(cand))


def run_monte_carlo(skills: np.ndarray, *, beta: float, dnf_rate: float,
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
    lo = mean_points - 1.96 * sd_points
    hi = mean_points + 1.96 * sd_points
    return {
        "p_title": titles / n_sim,
        "expected_points": mean_points,
        "points_lo95": lo,
        "points_hi95": hi,
        "expected_wins": wins_sum / n_sim,
    }


def top5_kendall(order_a: np.ndarray, order_b: np.ndarray, k: int = 5) -> float:
    """Kendall tau over the top-k ranks of two orderings (by index)."""
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


def negative_controls(skills: np.ndarray, *, beta: float, dnf_rate: float,
                      n_races: int, n_sim: int, seed: int) -> dict:
    out: dict = {}

    # (1) shift invariance: skill + constant -> identical title probabilities.
    #    Both runs use the same seed so the Gumbel stream is shared and only the
    #    location shifts, which cannot change any ranking.
    base = run_monte_carlo(skills, beta=beta, dnf_rate=dnf_rate,
                           n_races=n_races, n_sim=n_sim, seed=seed)
    shifted = run_monte_carlo(skills + 2.5, beta=beta, dnf_rate=dnf_rate,
                              n_races=n_races, n_sim=n_sim, seed=seed)
    delta = float(np.max(np.abs(shifted["p_title"] - base["p_title"])))
    out["shift_invariance"] = {
        "max_abs_delta_p_title": delta,
        "pass": bool(delta < 1e-9),
    }

    # (2) symmetry: two identical drivers in a 2-car field -> equal title odds.
    #    p0 + p1 == 1, so |p1 - p2| has std sqrt(1/n_sim); test against 3 sigma.
    sym = run_monte_carlo(
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


def sensitivity(skills: np.ndarray, *, beta_grid, dnf_grid, n_races, n_sim, seed) -> dict:
    base_order = np.argsort(
        -run_monte_carlo(skills, beta=float(np.median(beta_grid)),
                         dnf_rate=float(np.median(dnf_grid)),
                         n_races=n_races, n_sim=n_sim, seed=seed)["p_title"]
    )
    rows = []
    for b in beta_grid:
        for d in dnf_grid:
            res = run_monte_carlo(skills, beta=float(b), dnf_rate=float(d),
                                  n_races=n_races, n_sim=n_sim, seed=seed)
            order = np.argsort(-res["p_title"])
            rows.append({
                "beta": float(b),
                "dnf_rate": float(d),
                "top5_kendall_vs_base": top5_kendall(base_order, order),
            })
    return {"base_beta": float(np.median(beta_grid)), "base_dnf": float(np.median(dnf_grid)),
            "grid": rows}


# --------------------------------------------------------------------------- #
# Figure (best-effort)
# --------------------------------------------------------------------------- #
def render_figure(champions: pd.DataFrame, res: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - no matplotlib on some boxes
        print(f"[figure] skipped: {exc}")
        return

    df = champions.copy()
    df["p_title"] = res["p_title"]
    df["expected_points"] = res["expected_points"]
    df["points_lo"] = res["points_lo95"]
    df["points_hi"] = res["points_hi95"]
    df = df.sort_values("p_title", ascending=False).head(6)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    y = np.arange(len(df))[::-1]
    err = np.vstack([df["expected_points"] - df["points_lo"],
                     df["points_hi"] - df["expected_points"]])
    ax.barh(y, df["expected_points"], xerr=err, capsize=3,
            color="#d71920", alpha=0.85, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["driver_name"], fontsize=10)
    ax.set_xlabel("Expected season points (equal car, 95% CI)", fontsize=10)
    ax.set_title("Equal-car GOAT: expected points at each champion's peak",
                 fontsize=12, loc="left", pad=10)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.text(row["expected_points"] + 1.0, yi,
                f"P(title) {row['p_title']:.1%}", va="center", fontsize=8.5, color="#1a1c20")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Equal-car GOAT counterfactual")
    p.add_argument("--checkpoint", default="output/orthogonal_shapley_model/orthogonal_shapley.pth")
    p.add_argument("--meta", default="output/orthogonal_shapley_model/orthogonal_shapley_meta.json")
    p.add_argument("--baselines", default=None)
    p.add_argument("--max-year", type=int, default=2025)
    p.add_argument("--min-champ-year", type=int, default=1980)
    p.add_argument("--n-races", type=int, default=24)
    p.add_argument("--n-sim", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dnf-rate", type=float, default=0.05)
    p.add_argument("--output-dir", default="output/applications/goat_equal_car")
    args = p.parse_args()

    db = get_orthogonal_shapley_db()
    print("-> detecting champions and external peak seasons ...")
    champions = champion_peak_seasons(db, min_champ_year=args.min_champ_year)
    print(f"   {len(champions)} champions; example peak rows:\n"
          f"{champions.head().to_string(index=False)}")

    print("-> exporting skill scores from frozen model ...")
    export = load_season_skill(
        db, checkpoint=args.checkpoint, meta=args.meta,
        baselines=args.baselines, max_year=args.max_year,
    )
    skill = export.season[["driverId", "season", "skill_score"]].dropna()
    beta = calibrate_beta(export)
    print(f"   calibrated beta = {beta:.4f} (race-level residual std)")

    champions = champions.merge(
        skill, left_on=["driverId", "peak_season"], right_on=["driverId", "season"],
        how="left",
    )
    missing = champions[champions["skill_score"].isna()]
    if len(missing):
        print(f"   WARNING: {len(missing)} champions lack a peak-season skill (dropped):")
        print(missing[["driver_name", "peak_season"]].to_string(index=False))
    champions = champions.dropna(subset=["skill_score"]).reset_index(drop=True)
    skills = champions["skill_score"].to_numpy(dtype=float)

    print(f"-> Monte Carlo: {len(skills)} drivers, {args.n_races} races, "
          f"{args.n_sim} seasons, beta={beta:.4f}, dnf={args.dnf_rate}")
    res = run_monte_carlo(skills, beta=beta, dnf_rate=args.dnf_rate,
                          n_races=args.n_races, n_sim=args.n_sim, seed=args.seed)
    res["driver_name"] = champions["driver_name"].tolist()
    res["peak_season"] = champions["peak_season"].tolist()
    res["skill"] = skills.tolist()

    print("-> negative controls ...")
    controls = negative_controls(skills, beta=beta, dnf_rate=args.dnf_rate,
                                 n_races=args.n_races, n_sim=args.n_sim, seed=args.seed)

    print("-> sensitivity ...")
    beta_grid = [max(beta * f, 1e-3) for f in (0.5, 1.0, 2.0)]
    dnf_grid = [0.0, args.dnf_rate, 2 * args.dnf_rate]
    sens = sensitivity(skills, beta_grid=beta_grid, dnf_grid=dnf_grid,
                       n_races=args.n_races, n_sim=args.n_sim, seed=args.seed)

    ranking = pd.DataFrame({
        "driver_name": res["driver_name"],
        "peak_season": res["peak_season"],
        "skill": res["skill"],
        "p_title": res["p_title"],
        "expected_wins": res["expected_wins"],
        "expected_points": res["expected_points"],
    }).sort_values("p_title", ascending=False).reset_index(drop=True)

    payload = {
        "config": vars(args),
        "calibrated_beta": beta,
        "n_champions": int(len(champions)),
        "ranking": ranking.to_dict("records"),
        "negative_controls": controls,
        "sensitivity": sens,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "goat_equal_car.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)

    render_figure(ranking, res, out_dir / "goat_equal_car.png")

    print("\n=== Equal-car GOAT (top 6 by title probability) ===")
    print(ranking.head(6).to_string(index=False))
    print(f"\ncontrols: {json.dumps(controls, indent=2, default=float)}")
    print(f"wrote {out_dir / 'goat_equal_car.json'}")


if __name__ == "__main__":
    main()
