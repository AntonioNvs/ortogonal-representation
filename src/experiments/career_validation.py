"""Career validation runner for skill sources."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath("src"))

import config as cfg
import data.tasks as data_tasks
from baselines.skill_loader import load_skill_export, season_scores_for_career
from baselines.skill_gnn_skill import get_skill_gnn_db
from experiments.evaluate_skill_model import join_career
from validation.inference import (
    cluster_bootstrap_spearman,
    eligible_promotion_auroc,
    fisher_z_pooled,
    partial_spearman,
    permutation_within_season,
)
from validation.team_lineage import lineage_id_by_constructor
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers


def main() -> None:
    parser = argparse.ArgumentParser(description="Career validation for skill scores")
    parser.add_argument("--skill-source", type=str, default="bradley_terry")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to output/career_validation/{skill_source}",
    )
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default="output/skill_model/skill_gnn.pth")
    parser.add_argument("--meta", type=str, default="output/skill_model/skill_gnn_meta.json")
    args = parser.parse_args()

    source = "bayesian_ssm" if args.skill_source == "bayesian_comparator" else args.skill_source
    output_dir = args.output_dir or os.path.join("output/career_validation", args.skill_source)

    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )
    db = get_skill_gnn_db()
    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
    team_tier = compute_team_tiers(compute_constructor_season_points(db), lineage=lineage)

    export = load_skill_export(
        source,
        db,
        checkpoint_path=args.checkpoint,
        meta_path=args.meta,
    )
    skill = season_scores_for_career(export)
    joined = join_career(skill, db, team_tier, horizon=args.horizon)

    report = {
        "skill_source": args.skill_source,
        "n_rows": len(joined),
        "cluster_bootstrap": cluster_bootstrap_spearman(joined),
        "partial_spearman": partial_spearman(joined),
        "permutation": permutation_within_season(joined),
        "eligible_promotion_auroc": eligible_promotion_auroc(joined),
    }

    per_season = []
    for season, grp in joined.groupby("season_T"):
        if len(grp) < 5:
            continue
        from scipy.stats import spearmanr

        rho, _ = spearmanr(grp["skill_score"], grp["outcome_score"])
        per_season.append({"season": int(season), "n": len(grp), "spearman": float(rho)})
    report["fisher_z_pooled"] = fisher_z_pooled(
        __import__("pandas").DataFrame(per_season) if per_season else __import__("pandas").DataFrame()
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "correlation.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    joined.to_csv(os.path.join(output_dir, "joined.csv"), index=False)
    skill.to_csv(os.path.join(output_dir, "skill_scores.csv"), index=False)
    export.race.to_parquet(os.path.join(output_dir, "race_skill.parquet"), index=False)
    team_tier.to_csv(os.path.join(output_dir, "team_tiers.csv"), index=False)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
