"""Censored survival analysis for time-to-first-promotion.

The career join (`rest_of_career_outcome`) already carries the ingredients:
``seasons_to_promotion`` (event time), ``first_promotion_season`` (None = censored),
``n_future_seasons`` (censoring time), and ``peak_tier_score``. A binary
``promoted`` label discards time and misclassifies active-but-not-yet-promoted
drivers as never-promoted. This module models the censored time-to-event
directly, with no new runtime dependency (numpy/scipy only).

Semantics
---------
- Time origin = ``season_T``.
- Event = first season k > 0 with ``tier(T+k) > tier(T)``.
- Censoring time = ``n_future_seasons`` (active but not promoted at data cutoff).
- Ties handled by the Breslow approximation (Cox) / standard KM handling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _survival_inputs(
    df: pd.DataFrame,
    *,
    time_col: str = "seasons_to_promotion",
    event_col: str = "promoted",
    cens_time_col: str = "n_future_seasons",
    skill_col: str = "skill_score",
    cluster_col: str = "driverId",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, event, skill, cluster) aligned, with censored rows given
    time = censoring time and event = 0. ``event_col`` must already be the
    binary ``promoted`` flag (1 = observed promotion)."""
    sub = df.dropna(subset=[skill_col]).copy()
    time = sub[time_col].astype(float).to_numpy()
    event = sub[event_col].astype(float).to_numpy()
    # For censored rows, time_to_event is the observed follow-up.
    time = np.where(event == 1, time, sub[cens_time_col].astype(float).to_numpy())
    skill = sub[skill_col].astype(float).to_numpy()
    cluster = sub[cluster_col].to_numpy()
    return time, event, skill, cluster


def km_curve(
    time: np.ndarray,
    event: np.ndarray,
) -> dict:
    """Kaplan-Meier survival estimates.

    Returns ``times`` (unique event/censor times ascending), ``survival``
    (P(T > t)), and ``n_at_risk`` per time.
    """
    order = np.argsort(time, kind="stable")
    t = time[order]
    e = event[order]
    n = len(t)
    unique_times = np.unique(t)
    surv = []
    at_risk = []
    n_at = n
    j = 0
    s = 1.0
    for u in unique_times:
        n_at_risk = int((t >= u).sum())
        n_events = int(((t == u) & (e == 1)).sum())
        if n_at_risk > 0:
            s *= (1.0 - n_events / n_at_risk)
        surv.append(float(s))
        at_risk.append(n_at_risk)
    return {"times": unique_times.tolist(), "survival": surv, "n_at_risk": at_risk}


def km_by_tertile(
    time: np.ndarray,
    event: np.ndarray,
    skill: np.ndarray,
    *,
    n_tertiles: int = 3,
) -> dict:
    """Kaplan-Meier curves stratified by skill tertile.

    Returns a dict keyed by ``"bottom"``/``"mid"``/``"top"`` with each stratum's
    ``km_curve`` result plus its size and event count, and the tertile cut
    points. This is the stratified view the fair-market figure needs — the
    pooled ``km_curve`` alone cannot show that high-skill drivers promote
    faster.
    """
    labels = ["bottom", "mid", "top"][:n_tertiles]
    if np.unique(skill).size < n_tertiles or time.size < n_tertiles:
        return {"note": "insufficient skill spread for tertiles"}
    qs = np.quantile(skill, np.linspace(0, 1, n_tertiles + 1))
    out: dict = {"quantiles": qs.tolist()}
    for i, label in enumerate(labels):
        lo, hi = qs[i], qs[i + 1]
        if i == 0:
            mask = skill <= hi
        elif i == n_tertiles - 1:
            mask = skill >= lo
        else:
            mask = (skill > lo) & (skill <= hi)
        t_i = time[mask]
        e_i = event[mask]
        out[label] = {
            "km": km_curve(t_i, e_i),
            "n": int(mask.sum()),
            "n_events": int(e_i.sum()),
        }
    return out


def logrank_perm(
    time: np.ndarray,
    event: np.ndarray,
    group: np.ndarray,
    *,
    n_perm: int = 5000,
    seed: int = 0,
) -> dict:
    """Permutation log-rank test between two groups (group coded 0/1).

    Returns the observed log-rank chi-square statistic and a two-sided
    permutation p-value (shuffle group labels, recompute statistic).
    """
    mask = group.astype(float).astype(int)
    groups = np.unique(mask)
    if groups.size < 2:
        return {"chi2": float("nan"), "p_value": float("nan"), "note": "single group"}

    def _logrank(grp):
        g0 = grp == 0
        g1 = grp == 1
        unique_times = np.unique(time[event == 1])
        o1 = 0.0
        e1 = 0.0
        v1 = 0.0
        for u in unique_times:
            at_risk = time >= u
            if not at_risk.any():
                continue
            n_at = int(at_risk.sum())
            n_ev = int(((time == u) & (event == 1)).sum())
            n1_at = int((at_risk & g1).sum())
            if n_at <= 1:
                continue
            exp = n_ev * n1_at / n_at
            var = n_ev * (n_at - n_ev) * n1_at * (n_at - n1_at) / (n_at ** 2 * (n_at - 1))
            o1 += int(((time == u) & (event == 1) & g1).sum())
            e1 += exp
            v1 += var
        if v1 <= 0:
            return 0.0
        return (o1 - e1) ** 2 / v1

    chi2_obs = _logrank(mask)
    rng = np.random.default_rng(seed)
    ge = 0
    valid = 0
    for _ in range(n_perm):
        perm = rng.permutation(mask)
        if np.unique(perm).size < 2:
            continue
        stat = _logrank(perm)
        if np.isnan(stat):
            continue
        valid += 1
        if stat >= chi2_obs:
            ge += 1
    return {
        "chi2": float(chi2_obs),
        "p_value": float((ge + 1) / (valid + 1)) if valid else float("nan"),
        "n_perm_valid": valid,
    }


def cox_univariate(
    time: np.ndarray,
    event: np.ndarray,
    skill: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> dict:
    """Univariate Cox (Breslow partial likelihood) with skill as covariate.

    Returns log-hazard-ratio (beta), hazard ratio exp(beta), score-test p, and
    iteration count. Null model (beta=0) is the fit target via Newton-Raphson.
    """
    valid = np.isfinite(time) & np.isfinite(skill) & (time > 0)
    t = time[valid]
    e = event[valid]
    x = skill[valid]
    if t.size < 5 or (e == 1).sum() < 1 or np.std(x) == 0:
        return {"beta": float("nan"), "hazard_ratio": float("nan"),
                "p_value": float("nan"), "n": int(t.size), "n_events": int((e == 1).sum())}

    x = x - x.mean()
    beta = 0.0
    for _ in range(max_iter):
        # Breslow partial likelihood score and information. Risk-set weights use
        # the log-sum-exp (max-subtraction) trick so exp(beta*x) never overflows,
        # and the Newton step is damped so beta cannot diverge.
        score = 0.0
        info = 0.0
        for i in np.where(e == 1)[0]:
            risk = t >= t[i]
            xr = x[risk]
            w = np.exp(beta * xr - (beta * xr).max())  # stable softmax weights
            denom = w.sum()
            xbar = (xr * w).sum() / denom
            score += x[i] - xbar
            # information: weighted variance of x over the risk set
            info += ((xr ** 2 * w).sum() / denom) - xbar ** 2
        if info <= 1e-12:
            break
        step = score / info
        beta_new = beta + step
        halvings = 0
        while not np.isfinite(beta_new) and halvings < 30:
            step *= 0.5
            beta_new = beta + step
            halvings += 1
        beta = beta_new
        if abs(step) < tol:
            break

    # Score test at beta=0 for a model-agnostic p-value.
    score0 = 0.0
    info0 = 0.0
    for i in np.where(e == 1)[0]:
        risk = t >= t[i]
        xbar0 = x[risk].mean()
        score0 += x[i] - xbar0
        info0 += x[risk].var()
    from scipy.stats import chi2 as chi2_dist

    p = float(chi2_dist.sf(score0 ** 2 / info0, 1)) if info0 > 0 else float("nan")

    # Near-separation can push beta to a huge value; clamp to a finite range so
    # the HR stays JSON- and plot-safe (e^30 ≈ 1e13). Significance comes from the
    # score test below, not from the raw beta magnitude.
    if not np.isfinite(beta):
        beta = float("nan")
    else:
        beta = float(np.clip(beta, -30.0, 30.0))
    hr = float(np.exp(beta)) if np.isfinite(beta) else float("nan")

    return {
        "beta": beta,
        "hazard_ratio": hr,
        "p_value": p,
        "n": int(t.size),
        "n_events": int((e == 1).sum()),
    }


def cox_cluster_bootstrap_ci(
    time: np.ndarray,
    event: np.ndarray,
    skill: np.ndarray,
    cluster: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Cluster-bootstrap CI for the Cox hazard ratio (resample drivers)."""
    clusters = np.unique(cluster)
    idx_by_cluster = {c: np.where(cluster == c)[0] for c in clusters}
    rng = np.random.default_rng(seed)
    hrs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picks = rng.choice(clusters, size=clusters.size, replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        res = cox_univariate(time[idx], event[idx], skill[idx])
        hrs[b] = res["hazard_ratio"]
    valid = hrs[~np.isnan(hrs)]
    alpha = 1.0 - ci
    return {
        "hr_lo": float(np.quantile(valid, alpha / 2.0)) if valid.size else float("nan"),
        "hr_hi": float(np.quantile(valid, 1.0 - alpha / 2.0)) if valid.size else float("nan"),
        "n_valid_replicates": int(valid.size),
    }


def survival_analysis(
    df: pd.DataFrame,
    *,
    mask_col: str | None = "underrated_flag",
    skill_col: str = "skill_score",
    time_col: str = "seasons_to_promotion",
    event_col: str = "promoted",
    cens_time_col: str = "n_future_seasons",
    cluster_col: str = "driverId",
    seed: int = 0,
) -> dict:
    """Survival summary for one joined frame.

    ``mask_col`` restricts the analysis to a stratum (e.g. underrated); when None
    the eligible set is passed directly by the caller. Returns KM curve, a Cox
    fit of ``skill`` (continuous) with cluster-bootstrap HR CI, and a
    tertile-stratified log-rank test (top vs bottom skill tertile).
    """
    sub = df[df[mask_col]].copy() if mask_col else df.copy()
    time, event, skill, cluster = _survival_inputs(
        sub,
        time_col=time_col,
        event_col=event_col,
        cens_time_col=cens_time_col,
        skill_col=skill_col,
        cluster_col=cluster_col,
    )
    if time.size < 5:
        return {"n": int(time.size), "note": "insufficient rows"}

    km = km_curve(time, event)
    cox = cox_univariate(time, event, skill)
    hr_ci = cox_cluster_bootstrap_ci(time, event, skill, cluster, seed=seed)

    # Tertile stratification for the log-rank test.
    group = None
    logrank = {"note": "insufficient spread for tertiles"}
    if np.unique(skill).size >= 3:
        q = np.quantile(skill, [1 / 3, 2 / 3])
        low = skill <= q[0]
        high = skill >= q[1]
        if low.any() and high.any():
            idx = np.where(low | high)[0]
            group = (high.astype(int))[idx]
            logrank = logrank_perm(time[idx], event[idx], group, seed=seed)

    return {
        "n": int(time.size),
        "n_events": int(event.sum()),
        "n_censored": int((event == 0).sum()),
        "km": km,
        "km_tertiles": km_by_tertile(time, event, skill),
        "cox": {**cox, **hr_ci},
        "logrank_top_vs_bottom_tertile": logrank,
    }


def eligible_survival(
    df: pd.DataFrame,
    *,
    tier_col: str = "constructor_tier_score_at_T",
    top_tier_score: float = 3.0,
    **kwargs,
) -> dict:
    """Survival on all eligible drivers (tier at T below the top tier)."""
    sub = df[df[tier_col] < top_tier_score - 1e-9].copy()
    return survival_analysis(sub, mask_col=None, **kwargs)
