"""Career-validation runner.

Ties the deterministic framework together and reports how well a model's driver
skill score predicts forward career outcomes:

    1. tier teams S/A/B per season (deterministic, see ``validation.team_tiers``)
    2. build forward career labels (mean tier over T+1..T+horizon)
    3. load a per-(driver, season) skill score via a "skill scorer" adapter
    4. join skill vs. outcome and report Spearman/Kendall correlation

Model-agnostic: ``--skill-source`` selects the adapter; new architectures add a
scorer behind the same interface (a function returning ``[driverId, season,
skill_score]``).

Usage:
    python -m src.experiments.career_validation --skill-source kalman
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _p in (ROOT_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg
from data.enriched_dataset import EnrichedF1Dataset
from validation.career_labels import driver_season_constructor, forward_tier_outcome
from validation.inference import (
    cluster_bootstrap_spearman,
    fisher_z_pooled,
    moved_up_auroc,
    partial_spearman,
    permutation_within_season,
)
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers


def load_raw_db() -> object:
    """Raw enriched DB (full 2000-2026, no task, no ``upto`` truncation).

    Uses the same ``filter_db_by_years(2000, 2026)`` remapping as the Kalman
    graph, so ``driverId``/``constructorId`` match the model's id space.
    """
    return EnrichedF1Dataset().get_db(upto_test_timestamp=False)


def load_skill(skill_source: str, device, *, db=None, team_tier=None) -> pd.DataFrame:
    """Dispatch to a skill scorer by name. Returns [driverId, season, skill_score].

    Supported sources:
      * ``kalman``: readout of the trained Kalman-GNN pipeline.
      * ``points_share``: raw season-points share (naive market-aware baseline).
      * ``constructor_tier``: driver's constructor tier at T (adversarial
        baseline for the decomposition thesis). Requires ``team_tier``.
    """
    if skill_source == "kalman":
        from validation.kalman_skill import load_kalman_skill

        return load_kalman_skill(device=device)
    if skill_source == "points_share":
        from validation.baselines import load_points_share
        if db is None:
            db = load_raw_db()
        return load_points_share(db)
    if skill_source == "constructor_tier":
        from validation.baselines import load_constructor_tier
        if db is None or team_tier is None:
            raise ValueError("constructor_tier baseline requires db + team_tier.")
        return load_constructor_tier(db, team_tier)
    raise ValueError(
        f"Unknown --skill-source: {skill_source!r} "
        f"(supported: kalman, points_share, constructor_tier)"
    )


def bootstrap_spearman(x, y, n_bootstrap: int = 5000, ci: float = 0.95, seed: int = 0):
    """Naive row-bootstrap CI for Spearman's rho over paired (x, y).

    Kept for backwards compatibility and as a *diagnostic* — real inference
    on (driver, season) data should use
    :func:`validation.inference.cluster_bootstrap_spearman` since rows within
    a driver are not independent. The ``rho`` key here is the observed
    sample rho (not the bootstrap mean, which is a biased shrinkage
    estimator).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return {"rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(x.size)}

    rho_obs, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    rho_boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        r, _ = spearmanr(x[idx[i]], y[idx[i]])
        rho_boot[i] = r
    rho_boot = rho_boot[~np.isnan(rho_boot)]
    alpha = 1.0 - ci
    return {
        "rho": float(rho_obs),
        "boot_mean": float(rho_boot.mean()) if rho_boot.size else float("nan"),
        "ci_low": float(np.quantile(rho_boot, alpha / 2.0)) if rho_boot.size else float("nan"),
        "ci_high": float(np.quantile(rho_boot, 1.0 - alpha / 2.0)) if rho_boot.size else float("nan"),
        "n": int(x.size),
    }


def write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def run_career_validation(
    skill_source: str = "kalman",
    horizon: int | None = None,
    window: int | None = None,
    device=None,
    output_dir: str | None = None,
    min_year: int | None = None,
    n_bootstrap: int = 5000,
    n_perm: int = 20000,
    lineage: bool = False,
    require_full_horizon: bool = False,
):
    horizon = horizon or cfg.TIER_HORIZON
    window = window or cfg.TIER_WINDOW
    output_dir = output_dir or cfg.CAREER_VALIDATION_OUTPUT_DIR
    min_year = min_year or cfg.CAREER_VALIDATION_MIN_YEAR

    print("Loading raw DB...")
    db = load_raw_db()

    print("Computing deterministic team tiers...")
    points_df = compute_constructor_season_points(db)
    points_df = points_df[points_df["season"] >= min_year]
    p_S = cfg.TIER_S_FRAC
    p_A = cfg.TIER_A_FRAC
    lid_map = lineage_id_by_constructor(db.table_dict["constructors"].df) if lineage else None
    team_tier = compute_team_tiers(points_df, window=window, p_S=p_S, p_A=p_A, lineage=lid_map)
    print(f"  Tier proportions: S={p_S:.0%}, A={p_A:.0%}, B=remainder")
    print(f"  Lineage-aware tiers: {'yes' if lineage else 'no'}")

    # Ferrari sanity check (deterministic framework smoke signal).
    ferrari = team_tier[team_tier["constructorRef"].astype(str).str.lower() == "ferrari"]
    if not ferrari.empty:
        frac_s = (ferrari["tier"] == "S").mean()
        print(f"  Ferrari tier-S fraction: {frac_s:.2%} ({len(ferrari)} seasons)")

    print("Building forward career labels...")
    driver_season = driver_season_constructor(db)
    career_labels = forward_tier_outcome(
        driver_season, team_tier,
        horizon=horizon,
        require_full_horizon=require_full_horizon,
    )
    if require_full_horizon:
        print(f"  require_full_horizon=True -> kept only rows with n_observed >= {horizon}")

    print(f"Loading skill scores (source={skill_source})...")
    skill = load_skill(skill_source, device, db=db, team_tier=team_tier)

    # Join skill (driverId, season) <-> outcome (driverId, season_T).
    merged = skill.merge(
        career_labels,
        left_on=["driverId", "season"],
        right_on=["driverId", "season_T"],
        how="inner",
    ).dropna(subset=["skill_score", "outcome_score"])

    if merged.empty:
        print("No (driver, season) rows with both skill and outcome — aborting.")
        return

    # Annotate each row with the driver's constructor tier at T. Used for
    # partial-Spearman (skill | tier(T)) and AUROC "moved up".
    driver_season_cols = driver_season[["driverId", "season", "constructorId"]].rename(
        columns={"season": "season_T"}
    )
    merged = merged.merge(driver_season_cols, on=["driverId", "season_T"], how="left")
    tier_lookup = team_tier.set_index(["constructorId", "season"])["tier"]
    tier_score_at_T = []
    for cid, s in zip(merged["constructorId"], merged["season_T"]):
        try:
            t = tier_lookup.loc[(int(cid), int(s))]
            tier_score_at_T.append(float(cfg.TIER_TO_SCORE[t]))
        except (KeyError, ValueError, TypeError):
            tier_score_at_T.append(float("nan"))
    merged["tier_score_at_T"] = tier_score_at_T

    print(f"  Joined rows: {len(merged)} (drivers: {merged['driverId'].nunique()})")

    # --- Overall correlation ---
    rho, p_rho = spearmanr(merged["skill_score"], merged["outcome_score"])
    tau, p_tau = kendalltau(merged["skill_score"], merged["outcome_score"])

    # Naive row-bootstrap (kept as diagnostic — assumes iid rows, which is
    # violated because one driver contributes many correlated rows).
    rho_ci_naive = bootstrap_spearman(
        merged["skill_score"], merged["outcome_score"], n_bootstrap=n_bootstrap
    )

    # Honest CI: resample DRIVERS (not rows) with replacement. This is the
    # interval that should be quoted in the paper.
    rho_ci_cluster = cluster_bootstrap_spearman(
        merged, cluster_col="driverId", n_bootstrap=n_bootstrap
    )

    # --- Per-season Spearman ---
    per_season = []
    for season, g in merged.groupby("season"):
        if len(g) < 3:
            continue
        r, _ = spearmanr(g["skill_score"], g["outcome_score"])
        per_season.append({"season": int(season), "n": int(len(g)), "spearman": float(r)})
    per_season_df = pd.DataFrame(per_season)

    # Fisher-z pooled rho across seasons (within-season rows are much closer
    # to independent than the naive stack across seasons).
    rho_fisher = fisher_z_pooled(per_season_df) if not per_season_df.empty else {
        "rho_pooled": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
        "n_seasons": 0, "n_rows_total": 0,
    }

    # Permutation test conditional on season. Answers "does skill add signal
    # *within* a season, beyond what the grid already imposes?" — a tighter
    # null than the marginal spearmanr p-value.
    perm_result = permutation_within_season(merged, season_col="season_T", n_perm=n_perm)

    # Partial Spearman: skill vs. outcome, controlling for tier(T).
    # This is the sharpest test of the thesis — does the model add signal
    # above what "the driver already sits at a Tier-S team" gives you?
    partial_result = partial_spearman(
        merged,
        x_col="skill_score", y_col="outcome_score", z_col="tier_score_at_T",
        cluster_col="driverId", n_bootstrap=n_bootstrap,
    )

    # AUROC "did the driver move up a tier?" — interpretable effect size.
    auroc_result = moved_up_auroc(
        merged,
        skill_col="skill_score", outcome_col="outcome_score",
        baseline_col="tier_score_at_T", cluster_col="driverId",
        n_bootstrap=n_bootstrap,
    )

    # --- Persist ---
    os.makedirs(output_dir, exist_ok=True)

    write_csv(team_tier, os.path.join(output_dir, "team_tiers.csv"))
    write_csv(career_labels, os.path.join(output_dir, "career_labels.csv"))
    write_csv(skill, os.path.join(output_dir, "skill_scores.csv"))
    write_csv(merged, os.path.join(output_dir, "joined.csv"))
    write_csv(per_season_df, os.path.join(output_dir, "correlation_per_season.csv"))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skill_source": skill_source,
        "window": window,
        "horizon": horizon,
        "min_year": min_year,
        "lineage": lineage,
        "tier_s_frac": p_S,
        "tier_a_frac": p_A,
        "n_rows": int(len(merged)),
        "n_drivers": int(merged["driverId"].nunique()),
        # Observed sample statistic.
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),  # NAIVE p-value (iid); do NOT quote.
        "kendall_tau": float(tau),
        "kendall_p": float(p_tau),
        # Honest inference (paper-quality numbers):
        "cluster_bootstrap_spearman": rho_ci_cluster,
        "fisher_z_pooled": rho_fisher,
        "permutation_within_season": perm_result,
        "partial_spearman_given_tier_at_T": partial_result,
        "moved_up_auroc": auroc_result,
        # Diagnostic only:
        "naive_row_bootstrap": rho_ci_naive,
    }
    with open(os.path.join(output_dir, "correlation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    render_report(summary, per_season_df, output_dir)

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("CAREER VALIDATION")
    print("=" * 60)
    print(f"  Observed Spearman rho = {rho:+.4f}  (n={len(merged)}, drivers={merged['driverId'].nunique()})")
    print(f"  Kendall tau           = {tau:+.4f}")
    print(f"  Cluster-bootstrap 95% CI (by driverId) = "
          f"[{rho_ci_cluster['ci_low']:+.4f}, {rho_ci_cluster['ci_high']:+.4f}]")
    print(f"  Fisher-z pooled rho = {rho_fisher['rho_pooled']:+.4f}  "
          f"CI [{rho_fisher['ci_low']:+.4f}, {rho_fisher['ci_high']:+.4f}] over {rho_fisher['n_seasons']} seasons")
    print(f"  Within-season permutation p = {perm_result['p_value']:.4g}")
    print(f"  Partial rho | tier(T)  = {partial_result['partial_rho']:+.4f}  "
          f"CI [{partial_result['ci_low']:+.4f}, {partial_result['ci_high']:+.4f}]")
    print(f"  AUROC moved-up-tier    = {auroc_result['auroc']:.4f}  "
          f"CI [{auroc_result['ci_low']:.4f}, {auroc_result['ci_high']:.4f}]  "
          f"(n_pos={auroc_result['n_pos']}/{auroc_result['n_rows']})")
    print(f"\n  (naive iid p-value from spearmanr = {p_rho:.3g} — inflated, do not quote)")
    print(f"\n  Artifacts written to: {output_dir}")
    return summary


def render_report(summary: dict, per_season_df: pd.DataFrame, output_dir: str):
    cb = summary["cluster_bootstrap_spearman"]
    fz = summary["fisher_z_pooled"]
    perm = summary["permutation_within_season"]
    naive = summary["naive_row_bootstrap"]
    partial = summary["partial_spearman_given_tier_at_T"]
    auroc = summary["moved_up_auroc"]

    lines = [
        "# Career Validation Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Skill source: `{summary['skill_source']}`",
        f"- Tier window: `{summary['window']}` seasons (trailing)",
        f"- Forward horizon: `{summary['horizon']}` seasons",
        f"- Tier proportions: S=`{summary['tier_s_frac']:.0%}`, A=`{summary['tier_a_frac']:.0%}`, B=remainder",
        f"- Joined (driver, season) rows: `{summary['n_rows']}` across `{summary['n_drivers']}` drivers",
        "",
        "## Headline: skill vs. forward tier outcome",
        "",
        "| Statistic | Value | 95% CI | p |",
        "|---|---:|:---:|---:|",
        (f"| **Spearman rho** (cluster-bootstrap by driverId) | "
         f"**{summary['spearman_rho']:+.4f}** | "
         f"[{cb['ci_low']:+.4f}, {cb['ci_high']:+.4f}] | — |"),
        (f"| Fisher-z pooled rho (across {fz['n_seasons']} seasons) | "
         f"{fz['rho_pooled']:+.4f} | "
         f"[{fz['ci_low']:+.4f}, {fz['ci_high']:+.4f}] | — |"),
        (f"| Within-season permutation | rho={perm['rho_obs']:+.4f} | — | "
         f"**{perm['p_value']:.4g}** |"),
        f"| Kendall tau | {summary['kendall_tau']:+.4f} | — | {summary['kendall_p']:.3g} |",
        "",
        "## Effect size — does skill add signal *above* the driver's current team?",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---:|:---:|",
        (f"| **Partial Spearman rho** (controlling for tier(T)) | "
         f"**{partial['partial_rho']:+.4f}** | "
         f"[{partial['ci_low']:+.4f}, {partial['ci_high']:+.4f}] |"),
        (f"| **AUROC 'moved up a tier'** | "
         f"**{auroc['auroc']:.4f}** | "
         f"[{auroc['ci_low']:.4f}, {auroc['ci_high']:.4f}] |"),
        "",
        f"AUROC computed over {auroc['n_rows']} rows, "
        f"{auroc['n_pos']} positive (driver ended up at a strictly higher-tier team on average).",
        "",
        "**Interpretation.** The cluster-bootstrap CI is the honest interval — "
        "it treats each driver's sequence as one unit of information rather than "
        "one per (driver, season) row. The within-season permutation p-value tests "
        "whether skill orders drivers *within* a season, on top of what the grid "
        "already dictates. Partial rho is the sharpest test of the paper's thesis: "
        "if it stays positive with a CI not crossing zero, the skill score carries "
        "signal that 'the driver is at a Tier-S team right now' cannot explain.",
        "",
        "### Diagnostic (do not quote — assumes iid rows)",
        "",
        f"- Naive row-bootstrap 95% CI: [{naive['ci_low']:+.4f}, {naive['ci_high']:+.4f}]",
        f"- Naive iid p-value from scipy.stats.spearmanr: `{summary['spearman_p']:.3g}`",
        "",
        "## Per-Season Spearman",
        "",
        "| Season | n | Spearman rho |",
        "|---|---:|---:|",
    ]
    for _, row in per_season_df.iterrows():
        lines.append(f"| {int(row['season'])} | {int(row['n'])} | {row['spearman']:+.4f} |")

    path = os.path.join(output_dir, "career_validation_report.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Career validation: deterministic tiers + forward outcome vs. skill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skill-source", type=str, default="kalman", help="Skill scorer adapter (default: kalman).")
    parser.add_argument("--horizon", type=int, default=None, help="Forward horizon in seasons (default: cfg.TIER_HORIZON).")
    parser.add_argument("--window", type=int, default=None, help="Tier moving-average window (default: cfg.TIER_WINDOW).")
    parser.add_argument("--min-year", type=int, default=None, help="Earliest season (default: cfg.CAREER_VALIDATION_MIN_YEAR).")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. 'cuda:7', 'cpu').")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: cfg.CAREER_VALIDATION_OUTPUT_DIR).")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap resamples for rho CI (naive AND cluster).")
    parser.add_argument("--n-perm", type=int, default=20000, help="Permutations for within-season p-value.")
    parser.add_argument("--lineage", action="store_true", help="Make team tiers lineage-aware (rebrands carry their rank).")
    parser.add_argument("--require-full-horizon", action="store_true",
        help="Only keep rows with all forward seasons observed (recommended for the paper's headline numbers).")
    args = parser.parse_args()

    run_career_validation(
        skill_source=args.skill_source,
        horizon=args.horizon,
        window=args.window,
        device=args.device,
        output_dir=args.output_dir,
        min_year=args.min_year,
        n_bootstrap=args.n_bootstrap,
        n_perm=args.n_perm,
        lineage=args.lineage,
        require_full_horizon=args.require_full_horizon,
    )


if __name__ == "__main__":
    main()
