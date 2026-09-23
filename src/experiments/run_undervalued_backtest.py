"""Undervalued-driver backtest and prospective watchlist.

Two things, one frozen model:

1. **Retrospective frozen-model backtest** — for every historical season ``T``,
   compute each driver's "undervalued" score
       ``uv = z(skill_T) - z(car_strength_T)``
   where ``skill_T`` is the frozen model's season skill and ``car_strength_T`` is
   the driver's constructor points-share score (trailing 3-season mean), both
   standardized within season ``T``. Then measure whether ``uv`` predicts moving
   to a stronger team over the following ``--horizon`` seasons. The same frozen
   checkpoint is scored at every origin (not refit per season), so this is
   retrospective evidence for the watchlist rule, not a strict walk-forward
   refit estimate.

2. **Prospective watchlist** — the same ``uv`` score on eligible drivers only
   (below S-tier at ``T``, minimum race/sample experience, not a recent top-team
   seat-holder), on the most recent season and on retrospective 2024/2025 cuts.

Run on the A100:

    python src/experiments/run_undervalued_backtest.py \
      --checkpoint output/orthogonal_shapley_model/orthogonal_shapley.pth \
      --meta output/orthogonal_shapley_model/orthogonal_shapley_meta.json \
      --baselines output/orthogonal_shapley_model/coalition_baselines.json \
      --max-year 2026 --min-backtest-year 2000 --horizon 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo/src/

from baselines.orthogonal_shapley_skill import (  # noqa: E402
    export_orthogonal_shapley,
    get_orthogonal_shapley_db,
)
from utils.naming import build_driver_name_map  # noqa: E402
from validation.career_labels import driver_season_constructor  # noqa: E402
from validation.team_lineage import lineage_id_by_constructor  # noqa: E402
from validation.team_tiers import compute_constructor_season_points, compute_team_tiers  # noqa: E402


def _z_within_season(df: pd.DataFrame, col: str) -> pd.Series:
    """Within-season z-score; seasons with no variance return 0."""
    g = df.groupby("season")[col]
    mean = g.transform("mean")
    std = g.transform(lambda s: s.std(ddof=0))
    z = (df[col].astype(float) - mean) / std.replace(0.0, np.nan)
    return z.fillna(0.0)


def build_panel(db, *, checkpoint, meta, baselines, max_year, gpu_id=None) -> pd.DataFrame:
    """One row per (driverId, season) with skill, car strength, name, tier, age."""
    export = export_orthogonal_shapley(
        db, checkpoint_path=checkpoint, meta_path=meta,
        baselines_path=baselines, max_year=max_year, gpu_id=gpu_id,
    )
    skill = export.season[["driverId", "season", "skill_score"]].dropna(subset=["skill_score"])
    n_races = (
        export.race.groupby(["driverId", "season"])
        .size()
        .reset_index(name="n_races")
    )

    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
    tiers = compute_team_tiers(compute_constructor_season_points(db), lineage=lineage)
    car = tiers[["constructorId", "season", "score", "tier"]].rename(
        columns={"score": "car_strength"}
    )

    ds = driver_season_constructor(db)[["driverId", "driverRef", "season", "constructorId"]]
    drivers = db.table_dict["drivers"].df[["driverId", "dob"]].copy()
    drivers["dob"] = pd.to_datetime(drivers["dob"], errors="coerce")
    names = build_driver_name_map(db.table_dict["drivers"].df)

    panel = skill.merge(ds, on=["driverId", "season"], how="left")
    panel = panel.merge(car, on=["constructorId", "season"], how="left")
    panel = panel.merge(n_races, on=["driverId", "season"], how="left")
    panel = panel.merge(drivers, on="driverId", how="left")
    # Mid-season age (1 July of season year).
    mid = pd.to_datetime(panel["season"].astype(str) + "-07-01")
    panel["age"] = ((mid - panel["dob"]).dt.days / 365.25).astype(float)
    panel["driver_name"] = panel["driverId"].map(names)
    panel["n_races"] = panel["n_races"].fillna(0).astype(int)
    return panel.dropna(subset=["car_strength"]).reset_index(drop=True)


def _career_seasons_before(panel: pd.DataFrame, driver_id: int, season: int) -> int:
    return int(
        panel.loc[
            (panel["driverId"] == driver_id) & (panel["season"] < season),
            "season",
        ].nunique()
    )


def _had_s_tier_recently(
    panel: pd.DataFrame, driver_id: int, season: int, *, lookback: int,
) -> bool:
    if lookback <= 0:
        return False
    years = range(season - lookback, season)
    hist = panel[
        (panel["driverId"] == driver_id)
        & (panel["season"].isin(years))
        & (panel["tier"] == "S")
    ]
    return len(hist) > 0


def eligible_watchlist_rows(
    panel: pd.DataFrame,
    season: int,
    *,
    min_races: int,
    min_career_seasons: int,
    exclude_s_tier_within: int,
    max_age: float | None = None,
) -> pd.DataFrame:
    """Drivers who can plausibly move to a stronger team at season ``T``."""
    cur = panel[panel["season"] == season].copy()
    if cur.empty:
        return cur
    cur["uv"] = _z_within_season(cur, "skill_score") - _z_within_season(cur, "car_strength")
    cur["career_seasons_before"] = [
        _career_seasons_before(panel, int(d), season) for d in cur["driverId"]
    ]
    cur["recent_s_tier"] = [
        _had_s_tier_recently(panel, int(d), season, lookback=exclude_s_tier_within)
        for d in cur["driverId"]
    ]
    mask = (
        (cur["tier"] != "S")
        & (cur["n_races"] >= min_races)
        & (cur["career_seasons_before"] >= min_career_seasons)
        & (~cur["recent_s_tier"])
    )
    if max_age is not None and "age" in cur.columns:
        mask = mask & cur["age"].notna() & (cur["age"] <= float(max_age))
    return cur.loc[mask].sort_values("uv", ascending=False)


def rolling_backtest(
    panel: pd.DataFrame,
    *,
    min_year: int,
    horizon: int,
    top_k: int,
    min_races: int,
    min_career_seasons: int,
    exclude_s_tier_within: int,
    max_age: float | None = None,
) -> dict:
    """Rolling-origin evaluation of the undervalued score -> future promotion."""
    car_by_season = {
        (int(r.driverId), int(r.season)): float(r.car_strength)
        for r in panel.itertuples(index=False)
    }
    seasons = sorted(panel["season"].unique())
    max_season = int(max(seasons))

    rows = []
    for T in seasons:
        if T < min_year or T >= max_season:
            continue
        future_years = [T + k for k in range(1, horizon + 1) if T + k <= max_season]
        if not future_years:
            continue

        cur = eligible_watchlist_rows(
            panel,
            int(T),
            min_races=min_races,
            min_career_seasons=min_career_seasons,
            exclude_s_tier_within=exclude_s_tier_within,
            max_age=max_age,
        )
        if cur["car_strength"].nunique() < 2 or len(cur) < 5:
            continue

        future_peak = []
        for r in cur.itertuples(index=False):
            vals = [
                car_by_season.get((int(r.driverId), int(y)), np.nan) for y in future_years
            ]
            vals = [v for v in vals if np.isfinite(v)]
            future_peak.append(float(max(vals)) if vals else float(np.nan))
        cur = cur.copy()
        cur["future_peak_car"] = future_peak
        cur["promoted"] = cur["future_peak_car"] > cur["car_strength"] + 1e-9
        cur = cur.dropna(subset=["future_peak_car"])
        if cur["promoted"].nunique() < 2:
            continue

        base_rate = float(cur["promoted"].mean())
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(cur["promoted"].astype(int), cur["uv"]))
        except Exception:
            auroc = float("nan")

        k = min(top_k, len(cur))
        topk = cur.nlargest(k, "uv")
        precision = float(topk["promoted"].mean())
        topk_records = [
            {
                "driver_name": str(r.driver_name),
                "uv": float(r.uv),
                "tier": str(r.tier),
                "age": float(r.age) if pd.notna(getattr(r, "age", np.nan)) else None,
                "promoted": bool(r.promoted),
            }
            for r in topk.itertuples(index=False)
        ]
        rows.append({
            "season_T": int(T),
            "horizon_requested": int(horizon),
            "horizon_effective": int(len(future_years)),
            "future_years": future_years,
            "n_drivers": int(len(cur)),
            "base_promotion_rate": base_rate,
            "precision_at_k": precision,
            "lift": precision / base_rate if base_rate > 0 else float("nan"),
            "auroc": auroc,
            "topk_drivers": [r["driver_name"] for r in topk_records],
            "topk": topk_records,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return {"n_origins": 0, "rows": []}
    full_horizon = out[out["horizon_effective"] == horizon]
    summary_src = full_horizon if len(full_horizon) else out
    summary = {
        "n_origins": int(len(out)),
        "n_full_horizon_origins": int(len(full_horizon)),
        "mean_auroc": float(summary_src["auroc"].mean()),
        "mean_lift": float(summary_src["lift"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_base_rate": float(summary_src["base_promotion_rate"].mean()),
        "mean_precision_at_k": float(summary_src["precision_at_k"].mean()),
        "method": "retrospective_frozen_model",
        "note": (
            "Single frozen checkpoint scored at each historical origin T; not a "
            "walk-forward refit. Origins through max(season)-1 use partial future "
            f"windows when fewer than {horizon} post-T seasons exist; summary stats "
            f"prefer full-{horizon} origins."
        ),
    }
    return {"summary": summary, "rows": out.to_dict("records")}


def _watchlist_payload(
    panel: pd.DataFrame,
    season: int,
    *,
    top_k: int,
    min_races: int,
    min_career_seasons: int,
    exclude_s_tier_within: int,
    note: str,
    max_age: float | None = None,
) -> dict:
    cur = eligible_watchlist_rows(
        panel,
        season,
        min_races=min_races,
        min_career_seasons=min_career_seasons,
        exclude_s_tier_within=exclude_s_tier_within,
        max_age=max_age,
    )
    cols = [
        "driver_name", "driverRef", "constructorId", "tier",
        "skill_score", "car_strength", "n_races", "uv",
    ]
    if "age" in cur.columns:
        cols.append("age")
    return {
        "as_of_season": int(season),
        "note": note,
        "n_eligible": int(len(cur)),
        "watchlist": cur.head(top_k)[cols].to_dict("records"),
    }


def build_seasonal_screens(
    panel: pd.DataFrame,
    *,
    min_year: int,
    max_year: int,
    top_k: int,
    min_races: int,
    min_career_seasons: int,
    exclude_s_tier_within: int,
    max_age: float | None = None,
) -> list[dict]:
    """Per-season top undervalued drivers (eligible) from min_year..max_year."""
    avail = int(panel["season"].max())
    hi = min(max_year, avail)
    out: list[dict] = []
    for season in range(min_year, hi + 1):
        cur = eligible_watchlist_rows(
            panel,
            season,
            min_races=min_races,
            min_career_seasons=min_career_seasons,
            exclude_s_tier_within=exclude_s_tier_within,
            max_age=max_age,
        )
        if cur.empty:
            continue
        k = min(top_k, len(cur))
        topk = cur.nlargest(k, "uv")
        lead_age = topk.iloc[0]["age"] if "age" in topk.columns else np.nan
        out.append({
            "season": int(season),
            "n_eligible": int(len(cur)),
            "lead": {
                "driver_name": str(topk.iloc[0]["driver_name"]),
                "uv": float(topk.iloc[0]["uv"]),
                "tier": str(topk.iloc[0]["tier"]),
                "age": float(lead_age) if pd.notna(lead_age) else None,
            },
            "topk": [
                {
                    "driver_name": str(r.driver_name),
                    "uv": float(r.uv),
                    "tier": str(r.tier),
                    "age": float(r.age) if pd.notna(getattr(r, "age", np.nan)) else None,
                    "skill_score": float(r.skill_score),
                    "car_strength": float(r.car_strength),
                }
                for r in topk.itertuples(index=False)
            ],
        })
    return out


def build_watchlists(
    panel: pd.DataFrame,
    *,
    max_year: int,
    top_k: int,
    min_races: int,
    min_career_seasons: int,
    exclude_s_tier_within: int,
    retrospective_seasons: tuple[int, ...] = (2024, 2025),
    max_age: float | None = None,
) -> dict:
    """Prospective forecast plus retrospective eligible watchlists."""
    avail = int(panel["season"].max())
    forecast_season = min(max_year, avail)
    out = {
        "eligibility": {
            "tier_below_s": True,
            "min_races": min_races,
            "min_career_seasons": min_career_seasons,
            "exclude_s_tier_within": exclude_s_tier_within,
            "max_age": max_age,
        },
        "forecast": _watchlist_payload(
            panel,
            forecast_season,
            top_k=top_k,
            min_races=min_races,
            min_career_seasons=min_career_seasons,
            exclude_s_tier_within=exclude_s_tier_within,
            max_age=max_age,
            note=(
                "prospective forecast from frozen model; outcomes not yet observed"
            ),
        ),
        "retrospective": {},
    }
    for season in retrospective_seasons:
        if season > avail:
            continue
        out["retrospective"][str(season)] = _watchlist_payload(
            panel,
            season,
            top_k=top_k,
            min_races=min_races,
            min_career_seasons=min_career_seasons,
            exclude_s_tier_within=exclude_s_tier_within,
            max_age=max_age,
            note="retrospective eligible watchlist; outcomes known after season",
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Undervalued backtest + 2026 watchlist")
    p.add_argument("--checkpoint", default="output/orthogonal_shapley_model/orthogonal_shapley.pth")
    p.add_argument("--meta", default="output/orthogonal_shapley_model/orthogonal_shapley_meta.json")
    p.add_argument("--baselines", default=None)
    p.add_argument("--max-year", type=int, default=2026)
    p.add_argument("--min-backtest-year", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-races", type=int, default=5,
                   help="minimum races in season T for watchlist/backtest eligibility")
    p.add_argument("--min-career-seasons", type=int, default=0,
                   help="minimum prior F1 seasons before T (0 includes rookies)")
    p.add_argument("--exclude-s-tier-within", type=int, default=2,
                   help="exclude drivers with an S-tier seat in any of the prior N seasons")
    p.add_argument(
        "--max-age",
        type=float,
        default=25.0,
        help="max mid-season age for watchlists / seasonal screens (plot eligibility); "
             "set <=0 to disable",
    )
    p.add_argument("--output-dir", default="output/applications/undervalued")
    p.add_argument("--gpu-id", type=int, default=None,
                   help="CUDA device id (default: config.DEFAULT_GPU_ID)")
    args = p.parse_args()

    max_age = None if args.max_age is None or args.max_age <= 0 else float(args.max_age)

    db = get_orthogonal_shapley_db()
    print("-> building skill + car-strength panel ...")
    panel = build_panel(
        db, checkpoint=args.checkpoint, meta=args.meta,
        baselines=args.baselines, max_year=args.max_year,
        gpu_id=args.gpu_id,
    )
    print(f"   panel: {len(panel)} (driverId, season) rows; "
          f"seasons {panel['season'].min()}..{panel['season'].max()}")

    print("-> retrospective frozen-model backtest ...")
    backtest = rolling_backtest(
        panel,
        min_year=args.min_backtest_year,
        horizon=args.horizon,
        top_k=args.top_k,
        min_races=args.min_races,
        min_career_seasons=args.min_career_seasons,
        exclude_s_tier_within=args.exclude_s_tier_within,
        max_age=None,  # keep career AUROC on full eligible pool
    )
    print(f"   {backtest.get('summary', {}).get('n_origins', 0)} origins")
    if backtest.get("summary", {}).get("n_origins", 0):
        s = backtest["summary"]
        print(f"   mean AUROC {s['mean_auroc']:.3f} | mean precision@{args.top_k} "
              f"{s['mean_precision_at_k']:.3f} | mean lift {s['mean_lift']:.2f} "
              f"(base rate {s['mean_base_rate']:.3f})")

    age_note = f"age≤{max_age:.0f}" if max_age is not None else "no age cap"
    print(f"-> watchlists (forecast + retrospective; {age_note}) ...")
    watchlists = build_watchlists(
        panel,
        max_year=args.max_year,
        top_k=args.top_k,
        min_races=args.min_races,
        min_career_seasons=args.min_career_seasons,
        exclude_s_tier_within=args.exclude_s_tier_within,
        max_age=max_age,
    )

    print(f"-> seasonal screens (2014+; {age_note}) ...")
    seasonal_screens = build_seasonal_screens(
        panel,
        min_year=2014,
        max_year=args.max_year,
        top_k=args.top_k,
        min_races=args.min_races,
        min_career_seasons=args.min_career_seasons,
        exclude_s_tier_within=args.exclude_s_tier_within,
        max_age=max_age,
    )

    db_max_year = int(db.table_dict["races"].df["year"].max())
    payload = {
        "config": vars(args),
        "provenance": {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "db_max_year": db_max_year,
            "claim_level": "car_adjusted_performance",
            "repository": "https://github.com/AntonioNvs/ortogonal-representation",
        },
        "backtest": backtest,
        "watchlists": watchlists,
        "seasonal_screens": seasonal_screens,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "undervalued.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)

    print("\n=== Watchlist (prospective forecast) ===")
    for r in watchlists["forecast"]["watchlist"]:
        age = r.get("age")
        age_s = f" age={age:.1f}" if age is not None and np.isfinite(age) else ""
        print(f"  {r['driver_name']:<22} {str(r['tier']):<4} uv={r['uv']:+.2f} "
              f"(skill {r['skill_score']:+.2f}, car {r['car_strength']:.3f}){age_s}")
    print(f"\nwrote {out_dir / 'undervalued.json'}")


if __name__ == "__main__":
    main()
