"""Undervalued-driver backtest and prospective watchlist.

Two things, one frozen model:

1. **Rolling-origin backtest** — for every historical season ``T``, compute each
   driver's "undervalued" score
       ``uv = z(skill_T) - z(car_strength_T)``
   where ``skill_T`` is the model's season skill and ``car_strength_T`` is the
   driver's constructor points-share score (trailing 3-season mean), both
   standardized within season ``T``. Then measure whether ``uv`` predicts moving
   to a stronger team over the following ``--horizon`` seasons. This is the
   honest test: a watchlist procedure is only worth a 2026 forecast if it has
   beaten the base promotion rate historically.

2. **Prospective watchlist** — the same ``uv`` score computed on the most recent
   available season (``--max-year``, default 2026), reported as a *forecast*,
   never as a confirmed result.

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


def build_panel(db, *, checkpoint, meta, baselines, max_year) -> pd.DataFrame:
    """One row per (driverId, season) with skill, car strength, name, tier."""
    export = export_orthogonal_shapley(
        db, checkpoint_path=checkpoint, meta_path=meta,
        baselines_path=baselines, max_year=max_year,
    )
    skill = export.season[["driverId", "season", "skill_score"]].dropna(subset=["skill_score"])

    lineage = lineage_id_by_constructor(db.table_dict["constructors"].df)
    tiers = compute_team_tiers(compute_constructor_season_points(db), lineage=lineage)
    car = tiers[["constructorId", "season", "score", "tier"]].rename(
        columns={"score": "car_strength"}
    )

    ds = driver_season_constructor(db)[["driverId", "driverRef", "season", "constructorId"]]
    names = build_driver_name_map(db.table_dict["drivers"].df)

    panel = skill.merge(ds, on=["driverId", "season"], how="left")
    panel = panel.merge(car, on=["constructorId", "season"], how="left")
    panel["driver_name"] = panel["driverId"].map(names)
    return panel.dropna(subset=["car_strength"]).reset_index(drop=True)


def rolling_backtest(panel: pd.DataFrame, *, min_year: int, horizon: int) -> dict:
    """Rolling-origin evaluation of the undervalued score -> future promotion."""
    # future peak car strength within [T+1, T+horizon]
    car_by_season = {
        (int(r.driverId), int(r.season)): float(r.car_strength)
        for r in panel.itertuples(index=False)
    }
    seasons = sorted(panel["season"].unique())

    rows = []
    for T in seasons:
        if T < min_year or T > max(seasons) - horizon:
            continue
        cur = panel[panel["season"] == T].copy()
        if cur["car_strength"].nunique() < 2 or len(cur) < 5:
            continue
        cur["uv"] = _z_within_season(cur, "skill_score") - _z_within_season(cur, "car_strength")

        future_peak = []
        for r in cur.itertuples(index=False):
            vals = [car_by_season.get((int(r.driverId), int(T + k)), np.nan)
                    for k in range(1, horizon + 1)]
            vals = [v for v in vals if np.isfinite(v)]
            future_peak.append(float(max(vals)) if vals else float(np.nan))
        cur["future_peak_car"] = future_peak
        cur["promoted"] = cur["future_peak_car"] > cur["car_strength"] + 1e-9
        cur = cur.dropna(subset=["future_peak_car"])
        if cur["promoted"].nunique() < 2:
            continue

        base_rate = float(cur["promoted"].mean())
        # AUROC of uv on promoted
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(cur["promoted"].astype(int), cur["uv"]))
        except Exception:
            auroc = float("nan")

        k = min(5, len(cur))
        topk = cur.nlargest(k, "uv")
        precision = float(topk["promoted"].mean())
        rows.append({
            "season_T": int(T),
            "n_drivers": int(len(cur)),
            "base_promotion_rate": base_rate,
            "precision_at_k": precision,
            "lift": precision / base_rate if base_rate > 0 else float("nan"),
            "auroc": auroc,
            "topk_drivers": topk["driver_name"].tolist(),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return {"n_origins": 0, "rows": []}
    summary = {
        "n_origins": int(len(out)),
        "mean_auroc": float(out["auroc"].mean()),
        "mean_lift": float(out["lift"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_base_rate": float(out["base_promotion_rate"].mean()),
        "mean_precision_at_k": float(out["precision_at_k"].mean()),
    }
    return {"summary": summary, "rows": out.to_dict("records")}


def prospective_watchlist(panel: pd.DataFrame, *, max_year: int, top_k: int = 5) -> dict:
    """Undervalued ranking on the most recent available season (forecast)."""
    avail = panel["season"].max()
    season = min(max_year, avail)
    cur = panel[panel["season"] == season].copy()
    cur["uv"] = _z_within_season(cur, "skill_score") - _z_within_season(cur, "car_strength")
    cur = cur.sort_values("uv", ascending=False)
    return {
        "as_of_season": int(season),
        "requested_max_year": int(max_year),
        "note": "prospective forecast; not a confirmed outcome",
        "watchlist": cur.head(top_k)[
            ["driver_name", "driverRef", "constructorId", "tier",
             "skill_score", "car_strength", "uv"]
        ].to_dict("records"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Undervalued backtest + 2026 watchlist")
    p.add_argument("--checkpoint", default="output/orthogonal_shapley_model/orthogonal_shapley.pth")
    p.add_argument("--meta", default="output/orthogonal_shapley_model/orthogonal_shapley_meta.json")
    p.add_argument("--baselines", default=None)
    p.add_argument("--max-year", type=int, default=2026)
    p.add_argument("--min-backtest-year", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output-dir", default="output/applications/undervalued")
    args = p.parse_args()

    db = get_orthogonal_shapley_db()
    print("-> building skill + car-strength panel ...")
    panel = build_panel(
        db, checkpoint=args.checkpoint, meta=args.meta,
        baselines=args.baselines, max_year=args.max_year,
    )
    print(f"   panel: {len(panel)} (driverId, season) rows; "
          f"seasons {panel['season'].min()}..{panel['season'].max()}")

    print("-> rolling-origin backtest ...")
    backtest = rolling_backtest(panel, min_year=args.min_backtest_year,
                                horizon=args.horizon)
    print(f"   {backtest.get('summary', {}).get('n_origins', 0)} origins")
    if backtest.get("summary", {}).get("n_origins", 0):
        s = backtest["summary"]
        print(f"   mean AUROC {s['mean_auroc']:.3f} | mean precision@{args.top_k} "
              f"{s['mean_precision_at_k']:.3f} | mean lift {s['mean_lift']:.2f} "
              f"(base rate {s['mean_base_rate']:.3f})")

    print("-> prospective watchlist ...")
    watchlist = prospective_watchlist(panel, max_year=args.max_year, top_k=args.top_k)

    payload = {
        "config": vars(args),
        "backtest": backtest,
        "watchlist": watchlist,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "undervalued.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)

    print("\n=== Watchlist (prospective forecast) ===")
    for r in watchlist["watchlist"]:
        print(f"  {r['driver_name']:<22} {str(r['tier']):<4} uv={r['uv']:+.2f} "
              f"(skill {r['skill_score']:+.2f}, car {r['car_strength']:.3f})")
    print(f"\nwrote {out_dir / 'undervalued.json'}")


if __name__ == "__main__":
    main()
