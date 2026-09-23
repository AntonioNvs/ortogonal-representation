"""Helpers for equal-car GOAT counterfactuals (score-only and 2025 common-car)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


NEUTRAL_GRID = 10.5  # midfield common grid; not observed per-driver grids
F1_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)


def peak_driver_state_indices(
    res,
    *,
    driver_ids: list[int],
    peak_seasons: list[int],
) -> dict[int, int]:
    """Final causal ``driver_state`` index in each champion's peak season."""
    years = res.year.cpu().numpy()
    rounds = res.round.cpu().numpy()
    dids = res.driver_id.cpu().numpy()
    ds_idx = res.driver_state_idx.cpu().numpy()
    out: dict[int, int] = {}
    for did, peak in zip(driver_ids, peak_seasons):
        mask = (dids == did) & (years == peak) & (ds_idx >= 0)
        if not mask.any():
            continue
        # Last round of the peak season (highest round; tie-break highest state idx).
        cand = np.flatnonzero(mask)
        best = cand[np.lexsort((ds_idx[cand], rounds[cand]))[-1]]
        out[int(did)] = int(ds_idx[best])
    return out


def peak_career_indices(
    res,
    *,
    driver_ids: list[int],
) -> dict[int, int]:
    """Stable career embedding index per driver (any row for that driver)."""
    if not hasattr(res, "driver_career_idx"):
        return {}
    dids = res.driver_id.cpu().numpy()
    career = res.driver_career_idx.cpu().numpy()
    out: dict[int, int] = {}
    for did in driver_ids:
        mask = dids == did
        if not mask.any():
            continue
        out[int(did)] = int(career[np.flatnonzero(mask)[0]])
    return out


def race_ids_for_year(res, year: int) -> list[int]:
    """Sorted raceIds for a season (by round)."""
    years = res.year.cpu().numpy()
    rounds = res.round.cpu().numpy()
    race_ids = res.race_id.cpu().numpy()
    mask = years == year
    if not mask.any():
        return []
    df = pd.DataFrame(
        {
            "raceId": race_ids[mask],
            "round": rounds[mask],
        }
    ).drop_duplicates("raceId")
    return [int(r) for r in df.sort_values("round")["raceId"].tolist()]


def constructor_state_indices_for_race(res, race_id: int) -> np.ndarray:
    """Unique constructor_state indices present in a race."""
    race_ids = res.race_id.cpu().numpy()
    c_idx = res.constructor_state_idx.cpu().numpy()
    mask = (race_ids == race_id) & (c_idx >= 0)
    return np.unique(c_idx[mask]).astype(np.int64)


def mean_constructor_embedding(
    x_dict: dict[str, torch.Tensor],
    constructor_state_indices: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Mean constructor_state embedding over the given indices (common car)."""
    idx = torch.as_tensor(
        constructor_state_indices, device=x_dict["constructor_state"].device, dtype=torch.long
    )
    if idx.numel() == 0:
        raise ValueError("no constructor_state indices for mean car")
    return x_dict["constructor_state"][idx].mean(dim=0)


def race_meta_for_id(res, race_id: int) -> tuple[int, int, int]:
    """Return (race_idx, round, year) for a raceId (first matching results row)."""
    race_ids = res.race_id.cpu().numpy()
    mask = race_ids == race_id
    if not mask.any():
        raise ValueError(f"raceId {race_id} not found in results")
    i = int(np.flatnonzero(mask)[0])
    return (
        int(res.race_idx[i].item()),
        int(res.round[i].item()),
        int(res.year[i].item()),
    )


def neutral_grid_tensor(n: int, *, device: torch.device, value: float = NEUTRAL_GRID) -> torch.Tensor:
    return torch.full((n,), float(value), device=device, dtype=torch.float32)


@torch.no_grad()
def counterfactual_race_utilities(
    model,
    x_dict: dict[str, torch.Tensor],
    *,
    driver_state_indices: list[int],
    career_indices: list[int] | None,
    mean_constructor_emb: torch.Tensor,
    race_idx: int,
    round_num: int,
    grid_value: float = NEUTRAL_GRID,
) -> torch.Tensor:
    """Utilities for champions sharing one car + one race context.

    Each driver keeps their peak ``driver_state`` (and career) embedding; the
    constructor embedding is the shared mean car; context uses the target race
    with a neutral common grid.
    """
    device = mean_constructor_emb.device
    n = len(driver_state_indices)
    d_idx = torch.tensor(driver_state_indices, device=device, dtype=torch.long)
    d_emb = x_dict["driver_state"][d_idx]
    c_emb = mean_constructor_emb.unsqueeze(0).expand(n, -1).contiguous()
    race_t = torch.full((n,), int(race_idx), device=device, dtype=torch.long)
    round_t = torch.full((n,), float(round_num), device=device)
    grid = neutral_grid_tensor(n, device=device, value=grid_value)
    career_emb = None
    if career_indices is not None and model.driver_career is not None:
        career_t = torch.tensor(career_indices, device=device, dtype=torch.long)
        career_emb = model.driver_career(career_t)
    ctx = model.context_vector(x_dict, race_t, grid, round_t)
    if getattr(model, "use_additive_readout", False):
        return model.utility_additive(d_emb, c_emb, ctx, career_emb)
    fused = model.fused_input(d_emb, c_emb, ctx)
    return model.utility_from_fused(fused)


def simulate_season_from_utilities(
    race_utilities: np.ndarray,
    *,
    beta: float,
    dnf_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``race_utilities`` shape (n_races, n_drivers) -> (points, wins)."""
    n_races, n = race_utilities.shape
    total = np.zeros(n)
    wins = np.zeros(n)
    points_table = np.asarray(F1_POINTS, dtype=float)
    for r in range(n_races):
        u = race_utilities[r] + beta * rng.gumbel(0.0, 1.0, size=n)
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


def assert_common_car_invariants(
    *,
    race_ids: list[int],
    per_race_constructor_means: list[torch.Tensor],
    grids: list[torch.Tensor],
    grid_value: float = NEUTRAL_GRID,
) -> dict[str, Any]:
    """Deterministic checks for the 2025 common-car construction."""
    assert len(race_ids) == len(per_race_constructor_means) == len(grids)
    assert len(race_ids) > 0
    assert len(set(race_ids)) == len(race_ids)
    for g in grids:
        assert torch.allclose(g, torch.full_like(g, grid_value))
    # Each race has exactly one shared constructor vector (caller passes the mean).
    for m in per_race_constructor_means:
        assert m.ndim == 1
    return {
        "n_races": len(race_ids),
        "unique_race_ids": True,
        "neutral_grid": True,
        "one_mean_car_per_race": True,
        "pass": True,
    }
