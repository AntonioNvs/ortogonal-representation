"""Locked-window ranking metrics for validation benchmarks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from models.ranking_likelihood import batch_pl_nll


def _race_groups(race_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (utilities, ranks) per race sorted by finish position."""
    groups = []
    skill_col = "raw_skill" if "raw_skill" in race_df.columns else "skill_score"
    for _, grp in race_df.groupby("raceId"):
        grp = grp.sort_values("race_position_order" if "race_position_order" in grp.columns else "round")
        if "race_position_order" in grp.columns:
            ranks = grp["race_position_order"].astype(float).to_numpy()
        elif "position" in grp.columns:
            ranks = grp["position"].astype(float).to_numpy()
        else:
            continue
        utils = grp[skill_col].astype(float).to_numpy()
        if len(utils) >= 2:
            groups.append((utils, ranks))
    return groups


def race_pl_nll_and_pairwise(
    race_df: pd.DataFrame,
    *,
    test_years: tuple[int, ...] = (2024, 2025),
) -> Dict[str, float]:
    """True per-race Plackett-Luce NLL and pairwise accuracy on locked test window."""
    season_col = "season" if "season" in race_df.columns else "year"
    sub = race_df[race_df[season_col].isin(test_years)].copy()
    if sub.empty:
        return {"pl_nll": float("nan"), "pairwise_acc": float("nan"), "n_races": 0}

    utilities_list = []
    ranks_list = []
    correct = 0
    total = 0

    skill_col = "raw_skill" if "raw_skill" in sub.columns else "skill_score"
    for _, grp in sub.groupby("raceId"):
        if "race_position_order" in grp.columns:
            grp = grp.sort_values("race_position_order")
            ranks = grp["race_position_order"].astype(float).to_numpy()
        elif "position" in grp.columns:
            grp = grp.sort_values("position")
            ranks = grp["position"].astype(float).to_numpy()
        else:
            continue
        utils = grp[skill_col].astype(float).to_numpy()
        if len(utils) < 2:
            continue
        utilities_list.append(torch.tensor(utils, dtype=torch.float32))
        ranks_list.append(torch.tensor(ranks, dtype=torch.float32))
        n = len(utils)
        for i in range(n):
            for j in range(i + 1, n):
                if ranks[i] == ranks[j]:
                    continue
                total += 1
                if (ranks[i] < ranks[j] and utils[i] > utils[j]) or (
                    ranks[j] < ranks[i] and utils[j] > utils[i]
                ):
                    correct += 1

    pl_nll = float(batch_pl_nll(utilities_list, ranks_list).item()) if utilities_list else float("nan")
    pairwise_acc = correct / max(total, 1)
    return {
        "pl_nll": pl_nll,
        "pairwise_acc": float(pairwise_acc),
        "n_races": len(utilities_list),
        "n_pairs": int(total),
    }


def attach_race_positions(race_export: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Merge finish positions from canonical panel for PL metrics."""
    pos_cols = ["raceId", "driverId", "race_position_order"]
    if "race_position_order" not in panel.columns:
        pos_cols = ["raceId", "driverId", "position"]
        panel = panel.rename(columns={"position": "race_position_order"})
    meta = panel[pos_cols].drop_duplicates()
    return race_export.merge(meta, on=["raceId", "driverId"], how="left")
