"""Statistical inference primitives for the career-validation framework.

The career-validation join produces rows of shape ``(driverId, season_T,
skill_score, outcome_score)``. A single driver contributes many correlated
rows (Vettel: 15+; a debutante: 1). Applying ``scipy.stats.spearmanr`` to
those rows and reporting its p-value assumes iid observations, which is
false — the reported significance is inflated by roughly the average
per-driver cluster size.

This module supplies three honest alternatives, all keyed by ``driverId``:

    * ``cluster_bootstrap_spearman``: resample drivers (not rows) with
      replacement; keep each sampled driver's whole sequence; recompute rho.
      The resulting CI is a valid interval for the population rho.
    * ``fisher_z_pooled``: per-season rho -> Fisher z -> weighted average
      (weight = n_t - 3) -> tanh back. Within-season observations are much
      closer to independent than the naive stack.
    * ``permutation_within_season``: null model = "skill is unrelated to the
      outcome, *conditional on the season*". Permute ``skill_score`` within
      each ``season_T`` block and recompute rho. Preserves the marginal tier
      structure and the seasonal grid, so the p-value is not contaminated by
      era effects.

None of these replace ``spearmanr`` — the observed rho is still the sample
statistic. The runner keeps the naive bootstrap for backwards compatibility
but should surface the cluster/permutation numbers as primary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """``spearmanr`` that returns NaN instead of warning on constant input."""
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def cluster_bootstrap_spearman(
    df: pd.DataFrame,
    cluster_col: str = "driverId",
    x_col: str = "skill_score",
    y_col: str = "outcome_score",
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """CI for Spearman's rho where the resampling unit is a cluster (driver).

    In each iteration we sample ``n_clusters`` cluster ids with replacement,
    concatenate their whole row groups, and recompute rho. The bootstrap
    replicates that reproduce the *dependence structure* of the data, so the
    resulting percentile CI honours the fact that Vettel's 15 rows contribute
    a single unit of information, not 15.

    Returns dict with keys ``rho`` (observed rho on the full sample, NOT the
    bootstrap mean), ``ci_low``, ``ci_high``, ``n_clusters``, ``n_rows``.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    rho_obs = _safe_spearmanr(x, y)

    clusters = df[cluster_col].to_numpy()
    unique = np.unique(clusters)
    n_clusters = unique.size
    if n_clusters < 2:
        return {
            "rho": rho_obs,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_clusters": int(n_clusters),
            "n_rows": int(len(df)),
        }

    # Precompute per-cluster index arrays once.
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in unique}

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(unique, size=n_clusters, replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        replicates[b] = _safe_spearmanr(x[idx], y[idx])

    valid = replicates[~np.isnan(replicates)]
    alpha = 1.0 - ci
    if valid.size == 0:
        return {
            "rho": rho_obs,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_clusters": int(n_clusters),
            "n_rows": int(len(df)),
        }
    return {
        "rho": rho_obs,
        "ci_low": float(np.quantile(valid, alpha / 2.0)),
        "ci_high": float(np.quantile(valid, 1.0 - alpha / 2.0)),
        "boot_mean": float(valid.mean()),
        "boot_std": float(valid.std(ddof=1)),
        "n_clusters": int(n_clusters),
        "n_rows": int(len(df)),
        "n_valid_replicates": int(valid.size),
    }


def fisher_z_pooled(per_season_df: pd.DataFrame, ci: float = 0.95) -> dict:
    """Fisher-z pooled rho across seasons.

    Given a table of ``(season, n, spearman)``, transform ``z_t = atanh(rho_t)``,
    combine as a variance-weighted mean with weight ``n_t - 3`` (Fisher's
    variance-stabilising choice), and revert via ``tanh``. The pooled CI is
    ``tanh(z_bar +/- z_alpha/2 * sqrt(1/sum(w)))``.

    Ignores seasons with n<4 (Fisher's approximation degenerates).
    """
    df = per_season_df.copy()
    df = df[df["n"] >= 4].copy()
    df = df.dropna(subset=["spearman"])
    df = df[df["spearman"].abs() < 1.0]  # atanh(+/-1) diverges

    if df.empty:
        return {
            "rho_pooled": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_seasons": 0,
            "n_rows_total": 0,
        }

    z = np.arctanh(df["spearman"].to_numpy(dtype=float))
    w = (df["n"].to_numpy(dtype=float) - 3.0).clip(min=0.0)
    if w.sum() == 0.0:
        return {
            "rho_pooled": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_seasons": int(len(df)),
            "n_rows_total": int(df["n"].sum()),
        }

    z_bar = float(np.sum(w * z) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    alpha = 1.0 - ci
    from scipy.stats import norm

    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    lo, hi = z_bar - z_crit * se, z_bar + z_crit * se
    return {
        "rho_pooled": float(np.tanh(z_bar)),
        "ci_low": float(np.tanh(lo)),
        "ci_high": float(np.tanh(hi)),
        "z_bar": z_bar,
        "z_se": se,
        "n_seasons": int(len(df)),
        "n_rows_total": int(df["n"].sum()),
    }


def partial_spearman(
    df: pd.DataFrame,
    x_col: str = "skill_score",
    y_col: str = "outcome_score",
    z_col: str = "constructor_tier_score_at_T",
    cluster_col: str = "driverId",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Partial Spearman: rho(x, y | z) via residualisation on ranks.

    Steps:
        1. Rank-transform x, y, z.
        2. OLS-residualise each against z-rank.
        3. Pearson between the residuals -> partial rho.
        4. Cluster-bootstrap CI by ``cluster_col``.

    Answers the tightest version of the paper's question: does skill_score
    predict forward tier outcome *above and beyond* the driver's constructor
    tier at T? If yes, the model added value that raw "he's at a Tier-S team
    right now" cannot explain.
    """
    from scipy.stats import rankdata, pearsonr

    sub = df.dropna(subset=[x_col, y_col, z_col]).copy()
    if len(sub) < 5 or sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2 or sub[z_col].nunique() < 2:
        return {
            "partial_rho": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_rows": int(len(sub)),
        }

    def _partial(sub_df):
        rx = rankdata(sub_df[x_col].to_numpy())
        ry = rankdata(sub_df[y_col].to_numpy())
        rz = rankdata(sub_df[z_col].to_numpy())
        z_c = rz - rz.mean()
        denom = float(np.sum(z_c * z_c))
        if denom == 0.0:
            return float("nan")
        bx = float(np.sum(z_c * (rx - rx.mean())) / denom)
        by = float(np.sum(z_c * (ry - ry.mean())) / denom)
        ex = rx - bx * rz
        ey = ry - by * rz
        if np.std(ex) == 0 or np.std(ey) == 0:
            return float("nan")
        r, _ = pearsonr(ex, ey)
        return float(r)

    rho_obs = _partial(sub)

    clusters = sub[cluster_col].to_numpy()
    unique = np.unique(clusters)
    if unique.size < 2:
        return {"partial_rho": rho_obs, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_rows": int(len(sub)), "n_clusters": int(unique.size)}

    idx_by_cluster = {c: np.where(clusters == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    reps = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        reps[b] = _partial(sub.iloc[idx])
    reps = reps[~np.isnan(reps)]
    alpha = 1.0 - ci
    return {
        "partial_rho": rho_obs,
        "ci_low": float(np.quantile(reps, alpha / 2.0)) if reps.size else float("nan"),
        "ci_high": float(np.quantile(reps, 1.0 - alpha / 2.0)) if reps.size else float("nan"),
        "n_rows": int(len(sub)),
        "n_clusters": int(unique.size),
    }


def moved_up_auroc(
    df: pd.DataFrame,
    skill_col: str = "skill_score",
    outcome_col: str = "outcome_score",
    baseline_col: str = "constructor_tier_score_at_T",
    cluster_col: str = "driverId",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """AUROC of ``skill`` on the binary label ``outcome > baseline``.

    Concretely: for each (driver, T) row, label = 1 if the forward mean tier
    strictly exceeds the driver's tier at T (moved up), else 0. AUROC(skill,
    label) tests how well the skill score discriminates career promotions
    from non-promotions. Cluster-bootstrap CI by driver.

    NB: rows with ``outcome == baseline`` are neither positive nor negative;
    they still enter the ranking (as ties in the label), which is fine — the
    AUROC ranks positive over negative pairs and treats ties as neutral.
    """
    from sklearn.metrics import roc_auc_score

    sub = df.dropna(subset=[skill_col, outcome_col, baseline_col]).copy()
    sub["moved_up"] = (sub[outcome_col] > sub[baseline_col]).astype(int)

    if sub["moved_up"].nunique() < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_rows": int(len(sub)), "n_pos": int(sub["moved_up"].sum())}

    auroc_obs = float(roc_auc_score(sub["moved_up"].to_numpy(), sub[skill_col].to_numpy()))

    clusters = sub[cluster_col].to_numpy()
    unique = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    reps = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        y = sub["moved_up"].to_numpy()[idx]
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
        "n_pos": int(sub["moved_up"].sum()),
    }


def eligible_promotion_auroc(
    df: pd.DataFrame,
    skill_col: str = "skill_score",
    outcome_col: str = "outcome_score",
    baseline_col: str = "constructor_tier_score_at_T",
    top_tier_score: float = 3.0,
    cluster_col: str = "driverId",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """AUROC on drivers **eligible** to move up (below top tier at T).

    Top-tier drivers at T are excluded because they cannot promote further.
    Label: forward outcome tier strictly exceeds tier-at-T.
    """
    from sklearn.metrics import roc_auc_score

    sub = df.dropna(subset=[skill_col, outcome_col, baseline_col]).copy()
    sub = sub[sub[baseline_col] < top_tier_score - 1e-9].copy()
    sub["moved_up"] = (sub[outcome_col] > sub[baseline_col]).astype(int)

    if len(sub) < 5 or sub["moved_up"].nunique() < 2:
        return {
            "auroc": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_rows": int(len(sub)),
            "n_pos": int(sub["moved_up"].sum()) if len(sub) else 0,
            "note": "insufficient eligible rows or single-class labels",
        }

    auroc_obs = float(roc_auc_score(sub["moved_up"].to_numpy(), sub[skill_col].to_numpy()))
    clusters = sub[cluster_col].to_numpy()
    unique = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    reps = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        y = sub["moved_up"].to_numpy()[idx]
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
        "n_pos": int(sub["moved_up"].sum()),
        "note": "eligible drivers only (tier_at_T < S)",
    }


def permutation_within_season(
    df: pd.DataFrame,
    season_col: str = "season_T",
    x_col: str = "skill_score",
    y_col: str = "outcome_score",
    n_perm: int = 20000,
    seed: int = 0,
) -> dict:
    """Two-sided permutation test with within-season shuffling.

    H0: within each season, ``skill_score`` is exchangeable with respect to
    ``outcome_score`` — i.e. the model's skill ordering carries no forward
    signal beyond what the season's grid already dictates. We shuffle
    ``skill_score`` *inside* each season block, recompute rho on the full
    stack, and count how often the permuted |rho| meets or exceeds |rho_obs|.

    The choice of within-season shuffling matters: shuffling globally would
    destroy era effects and reject the null even for a model that only
    encodes "which year this driver drove". This design tests the tighter
    claim that skill *adds* signal within a season.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    rho_obs = _safe_spearmanr(x, y)

    seasons = df[season_col].to_numpy()
    unique = np.unique(seasons)
    block_idx = [np.where(seasons == s)[0] for s in unique]

    rng = np.random.default_rng(seed)
    ge_count = 0
    valid = 0
    for _ in range(n_perm):
        x_perm = x.copy()
        for idx in block_idx:
            if idx.size > 1:
                x_perm[idx] = rng.permutation(x_perm[idx])
        r = _safe_spearmanr(x_perm, y)
        if np.isnan(r):
            continue
        valid += 1
        if abs(r) >= abs(rho_obs):
            ge_count += 1

    if valid == 0:
        return {"rho_obs": rho_obs, "p_value": float("nan"), "n_perm": int(n_perm)}
    return {
        "rho_obs": float(rho_obs),
        "p_value": float((ge_count + 1) / (valid + 1)),  # add-one smoothing
        "n_perm_valid": int(valid),
    }
