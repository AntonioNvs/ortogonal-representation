"""Underrated-driver inconsistency detection and resolution metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def mark_underrated(
    df: pd.DataFrame,
    *,
    skill_col: str = "skill_score",
    season_col: str = "season_T",
    tier_col: str = "constructor_tier_score_at_T",
    skill_pct_threshold: float = 0.75,
    max_tier_score: float = 1.0,
) -> pd.DataFrame:
    """Flag drivers with high within-season skill but B-tier team at T."""
    out = df.copy()
    out["skill_percentile"] = out.groupby(season_col)[skill_col].rank(pct=True, method="average")
    out["underrated_flag"] = (
        (out["skill_percentile"] >= skill_pct_threshold)
        & (out[tier_col] <= max_tier_score + 1e-9)
    )
    out["promoted"] = (out["outcome_score"] > out[tier_col]).astype(int)
    return out


def _cluster_bootstrap_rate(
    df: pd.DataFrame,
    *,
    label_col: str,
    cluster_col: str = "driverId",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    rate_obs = float(df[label_col].mean()) if len(df) else float("nan")
    clusters = df[cluster_col].to_numpy()
    unique = np.unique(clusters)
    if unique.size < 2 or len(df) == 0:
        return {
            "rate": rate_obs,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_rows": int(len(df)),
            "n_clusters": int(unique.size),
        }

    idx_by_cluster = {c: np.where(clusters == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    reps = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        reps[b] = float(df[label_col].to_numpy()[idx].mean())
    alpha = 1.0 - ci
    return {
        "rate": rate_obs,
        "ci_low": float(np.quantile(reps, alpha / 2.0)),
        "ci_high": float(np.quantile(reps, 1.0 - alpha / 2.0)),
        "n_rows": int(len(df)),
        "n_clusters": int(unique.size),
    }


def underrated_resolution_rate(
    df: pd.DataFrame,
    *,
    mask_col: str = "underrated_flag",
    label_col: str = "promoted",
    cluster_col: str = "driverId",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Fraction of underrated rows eventually promoted (rest-of-career)."""
    sub = df[df[mask_col]].copy()
    if sub.empty:
        return {
            "resolution_rate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_underrated": 0,
            "n_promoted": 0,
        }
    boot = _cluster_bootstrap_rate(
        sub,
        label_col=label_col,
        cluster_col=cluster_col,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
    )
    return {
        "resolution_rate": boot["rate"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "n_underrated": int(len(sub)),
        "n_promoted": int(sub[label_col].sum()),
        "n_clusters": boot["n_clusters"],
    }


def underrated_promotion_auroc(
    df: pd.DataFrame,
    *,
    mask_col: str = "underrated_flag",
    skill_col: str = "skill_score",
    label_col: str = "promoted",
    cluster_col: str = "driverId",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """AUROC(skill, promoted) restricted to the underrated stratum."""
    sub = df[df[mask_col]].dropna(subset=[skill_col, label_col]).copy()
    if len(sub) < 5 or sub[label_col].nunique() < 2:
        return {
            "auroc": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_rows": int(len(sub)),
            "n_pos": int(sub[label_col].sum()) if len(sub) else 0,
        }

    auroc_obs = float(roc_auc_score(sub[label_col].to_numpy(), sub[skill_col].to_numpy()))
    clusters = sub[cluster_col].to_numpy()
    unique = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    reps = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        y = sub[label_col].to_numpy()[idx]
        s = sub[skill_col].to_numpy()[idx]
        if len(np.unique(y)) < 2:
            reps[b] = np.nan
            continue
        reps[b] = roc_auc_score(y, s)
    reps = reps[~np.isnan(reps)]
    alpha = 1.0 - ci
    return {
        "auroc": auroc_obs,
        "ci_low": float(np.quantile(reps, alpha / 2.0)) if reps.size else float("nan"),
        "ci_high": float(np.quantile(reps, 1.0 - alpha / 2.0)) if reps.size else float("nan"),
        "n_rows": int(len(sub)),
        "n_pos": int(sub[label_col].sum()),
    }


def time_to_promotion_summary(
    df: pd.DataFrame,
    *,
    mask_col: str = "underrated_flag",
    seasons_col: str = "seasons_to_promotion",
) -> dict:
    """Median seasons until first tier promotion among underrated drivers."""
    sub = df[df[mask_col]].copy()
    promoted = sub[sub["promoted"] == 1].dropna(subset=[seasons_col])
    if promoted.empty:
        return {
            "median_seasons": float("nan"),
            "mean_seasons": float("nan"),
            "n_promoted": 0,
            "n_underrated": int(len(sub)),
        }
    vals = promoted[seasons_col].astype(float)
    return {
        "median_seasons": float(vals.median()),
        "mean_seasons": float(vals.mean()),
        "n_promoted": int(len(promoted)),
        "n_underrated": int(len(sub)),
    }


def resolution_by_decade(
    df: pd.DataFrame,
    *,
    season_col: str = "season_T",
    mask_col: str = "underrated_flag",
    label_col: str = "promoted",
) -> pd.DataFrame:
    """Resolution rate stratified by decade of season_T."""
    sub = df[df[mask_col]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["decade", "resolution_rate", "n_underrated", "n_promoted"])
    sub["decade"] = (sub[season_col] // 10) * 10
    rows = []
    for decade, grp in sub.groupby("decade", sort=True):
        rows.append(
            {
                "decade": int(decade),
                "resolution_rate": float(grp[label_col].mean()),
                "n_underrated": int(len(grp)),
                "n_promoted": int(grp[label_col].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_inconsistency_report(
    df: pd.DataFrame,
    *,
    skill_pct_threshold: float = 0.75,
    max_tier_score: float = 1.0,
    seed: int = 0,
) -> dict:
    """Full inconsistency report for one skill source join."""
    marked = mark_underrated(
        df,
        skill_pct_threshold=skill_pct_threshold,
        max_tier_score=max_tier_score,
    )
    return {
        "underrated_resolution": underrated_resolution_rate(marked, seed=seed),
        "underrated_promotion_auroc": underrated_promotion_auroc(marked, seed=seed),
        "time_to_promotion": time_to_promotion_summary(marked),
        "resolution_by_decade": resolution_by_decade(marked).to_dict(orient="records"),
        "n_total_rows": int(len(marked)),
        "n_underrated": int(marked["underrated_flag"].sum()),
        "marked_df": marked,
    }
