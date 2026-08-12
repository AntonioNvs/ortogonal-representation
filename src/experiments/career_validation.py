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
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers


def load_raw_db() -> object:
    """Raw enriched DB (full 2000-2026, no task, no ``upto`` truncation).

    Uses the same ``filter_db_by_years(2000, 2026)`` remapping as the Kalman
    graph, so ``driverId``/``constructorId`` match the model's id space.
    """
    return EnrichedF1Dataset().get_db(upto_test_timestamp=False)


def load_skill(skill_source: str, device) -> pd.DataFrame:
    """Dispatch to a skill scorer by name. Returns [driverId, season, skill_score]."""
    if skill_source == "kalman":
        from validation.kalman_skill import load_kalman_skill

        return load_kalman_skill(device=device)
    raise ValueError(f"Unknown --skill-source: {skill_source!r} (supported: kalman)")


def bootstrap_spearman(x, y, n_bootstrap: int = 5000, ci: float = 0.95, seed: int = 0):
    """Bootstrap percentile CI for Spearman's rho over paired (x, y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return {"rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(x.size)}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    rho_boot = []
    for i in range(n_bootstrap):
        rho, _ = spearmanr(x[idx[i]], y[idx[i]])
        rho_boot.append(rho)
    rho_boot = np.asarray(rho_boot)
    alpha = 1.0 - ci
    return {
        "rho": float(rho_boot.mean()),
        "ci_low": float(np.quantile(rho_boot, alpha / 2.0)),
        "ci_high": float(np.quantile(rho_boot, 1.0 - alpha / 2.0)),
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
    lineage: bool = False,
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
    career_labels = forward_tier_outcome(driver_season, team_tier, horizon=horizon)

    print(f"Loading skill scores (source={skill_source})...")
    skill = load_skill(skill_source, device)

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

    print(f"  Joined rows: {len(merged)} (drivers: {merged['driverId'].nunique()})")

    # --- Overall correlation ---
    rho, p_rho = spearmanr(merged["skill_score"], merged["outcome_score"])
    tau, p_tau = kendalltau(merged["skill_score"], merged["outcome_score"])
    rho_ci = bootstrap_spearman(
        merged["skill_score"], merged["outcome_score"], n_bootstrap=n_bootstrap
    )

    # --- Per-season Spearman ---
    per_season = []
    for season, g in merged.groupby("season"):
        if len(g) < 3:
            continue
        r, _ = spearmanr(g["skill_score"], g["outcome_score"])
        per_season.append({"season": int(season), "n": int(len(g)), "spearman": float(r)})
    per_season_df = pd.DataFrame(per_season)

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
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "kendall_tau": float(tau),
        "kendall_p": float(p_tau),
        "spearman_bootstrap": rho_ci,
    }
    with open(os.path.join(output_dir, "correlation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    render_report(summary, per_season_df, output_dir)

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("CAREER VALIDATION")
    print("=" * 60)
    print(f"  Spearman rho  = {rho:+.4f} (p={p_rho:.3g}, n={len(merged)})")
    print(f"  Kendall tau   = {tau:+.4f} (p={p_tau:.3g})")
    print(f"  rho 95% CI    = [{rho_ci['ci_low']:.4f}, {rho_ci['ci_high']:.4f}]")
    print(f"\n  Artifacts written to: {output_dir}")
    return summary


def render_report(summary: dict, per_season_df: pd.DataFrame, output_dir: str):
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
        "## Overall Correlation (skill vs. forward outcome)",
        "",
        "| Metric | Value | p-value |",
        "|---|---:|---:|",
        f"| Spearman rho | {summary['spearman_rho']:+.4f} | {summary['spearman_p']:.3g} |",
        f"| Kendall tau | {summary['kendall_tau']:+.4f} | {summary['kendall_p']:.3g} |",
        "",
        f"Spearman rho bootstrap 95% CI: "
        f"[{summary['spearman_bootstrap']['ci_low']:.4f}, "
        f"{summary['spearman_bootstrap']['ci_high']:.4f}]",
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
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap resamples for rho CI.")
    parser.add_argument("--lineage", action="store_true", help="Make team tiers lineage-aware (rebrands carry their rank).")
    args = parser.parse_args()

    run_career_validation(
        skill_source=args.skill_source,
        horizon=args.horizon,
        window=args.window,
        device=args.device,
        output_dir=args.output_dir,
        min_year=args.min_year,
        n_bootstrap=args.n_bootstrap,
        lineage=args.lineage,
    )


if __name__ == "__main__":
    main()
