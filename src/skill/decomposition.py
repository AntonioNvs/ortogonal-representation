"""Normalized Shapley variance decomposition for driver / constructor / context."""

from __future__ import annotations

from itertools import permutations
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


COMPONENTS = ("driver", "constructor", "context")


def _subset_variance(values: np.ndarray, mask: np.ndarray) -> float:
    sub = values[mask]
    if sub.size < 2:
        return 0.0
    return float(np.var(sub, ddof=1))


def shapley_variance_shares(
    driver: np.ndarray,
    constructor: np.ndarray,
    context: np.ndarray,
) -> Dict[str, float]:
    """Exact Shapley allocation of Var(y) across three additive channels.

    y_i = driver_i + constructor_i + context_i (model systematic predictor).
    Residual / chance is excluded and reported separately by callers.
    """
    d = np.asarray(driver, dtype=float)
    c = np.asarray(constructor, dtype=float)
    x = np.asarray(context, dtype=float)
    y = d + c + x
    n = y.size
    if n < 2:
        return {k: 1.0 / 3.0 for k in COMPONENTS}

    total_var = float(np.var(y, ddof=1))
    if total_var <= 1e-12:
        return {k: 1.0 / 3.0 for k in COMPONENTS}

    channels = {"driver": d, "constructor": c, "context": x}
    phi = {k: 0.0 for k in COMPONENTS}

    for player in COMPONENTS:
        others = [k for k in COMPONENTS if k != player]
        for order in permutations(others):
            coalition: List[str] = []
            for step, member in enumerate([player] + list(order)):
                if step == 0:
                    coalition = [member]
                    continue
                prev = coalition.copy()
                coalition.append(member)
                v_with = _var_of_sum(channels, coalition, n)
                v_without = _var_of_sum(channels, prev, n)
                weight = 1.0 / len(others)  # uniform over permutations of others
                phi[member] += weight * max(v_with - v_without, 0.0)

    raw_sum = sum(phi.values())
    if raw_sum <= 1e-12:
        return {k: 1.0 / 3.0 for k in COMPONENTS}
    return {k: float(phi[k] / raw_sum) for k in COMPONENTS}


def _var_of_sum(channels: Dict[str, np.ndarray], keys: Iterable[str], n: int) -> float:
    s = sum(channels[k] for k in keys)
    if n < 2:
        return 0.0
    return float(np.var(s, ddof=1))


def aggregate_season_shapley(
    race_df: pd.DataFrame,
    *,
    driver_col: str = "contrib_driver",
    constructor_col: str = "contrib_constructor",
    context_col: str = "contrib_context",
    group_cols: Tuple[str, ...] = ("driverId", "season"),
) -> pd.DataFrame:
    """Per-(driver, season) mean Shapley shares with bootstrap CI by driver cluster."""
    rows = []
    for keys, grp in race_df.groupby(list(group_cols)):
        shares = shapley_variance_shares(
            grp[driver_col].to_numpy(),
            grp[constructor_col].to_numpy(),
            grp[context_col].to_numpy(),
        )
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({f"share_{k}": v for k, v in shares.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_shapley_ci(
    race_df: pd.DataFrame,
    *,
    driver_col: str = "contrib_driver",
    constructor_col: str = "contrib_constructor",
    context_col: str = "contrib_context",
    group_cols: Tuple[str, ...] = ("driverId", "season"),
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Cluster-bootstrap Shapley driver share CI by resampling races within group."""
    rng = np.random.default_rng(seed)
    base = aggregate_season_shapley(
        race_df,
        driver_col=driver_col,
        constructor_col=constructor_col,
        context_col=context_col,
        group_cols=group_cols,
    )
    if base.empty:
        return base

    reps: Dict[tuple, List[float]] = {}
    for keys, grp in race_df.groupby(list(group_cols)):
        key = keys if isinstance(keys, tuple) else (keys,)
        idx = np.arange(len(grp))
        for _ in range(n_bootstrap):
            pick = rng.choice(idx, size=len(idx), replace=True)
            sub = grp.iloc[pick]
            sh = shapley_variance_shares(
                sub[driver_col].to_numpy(),
                sub[constructor_col].to_numpy(),
                sub[context_col].to_numpy(),
            )
            reps.setdefault(key, []).append(sh["driver"])

    alpha = 1.0 - ci
    lo_map = {}
    hi_map = {}
    for key, vals in reps.items():
        arr = np.asarray(vals)
        lo_map[key] = float(np.quantile(arr, alpha / 2.0))
        hi_map[key] = float(np.quantile(arr, 1.0 - alpha / 2.0))

    out = base.copy()
    out["share_driver_lo"] = [
        lo_map.get(tuple(row[c] for c in group_cols), float("nan"))
        for _, row in out.iterrows()
    ]
    out["share_driver_hi"] = [
        hi_map.get(tuple(row[c] for c in group_cols), float("nan"))
        for _, row in out.iterrows()
    ]
    return out
