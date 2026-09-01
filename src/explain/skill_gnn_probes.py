"""XAI falsification probes for SkillGNN on the temporal graph."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from baselines.skill_gnn_skill import load_skill_gnn_model_and_graph
from models.skill_gnn import SkillGNN
from relbench.base import Database


@dataclass
class ProbeSampleConfig:
    min_year: int = 2024
    max_year: int = 2025
    max_samples: int = 1000
    swap_samples: int = 200
    seed: int = 42


def load_skill_gnn_for_probes(
    db: Database,
    checkpoint_path: str = "output/skill_model/skill_gnn.pth",
    meta_path: str = "output/skill_model/skill_gnn_meta.json",
):
    """Load SkillGNN checkpoint and causal graph tensors for probe inference."""
    return load_skill_gnn_model_and_graph(
        db, checkpoint_path=checkpoint_path, meta_path=meta_path
    )


def sample_race_rows(
    graph_data,
    config: ProbeSampleConfig,
) -> torch.Tensor:
    """Sample result row indices from the test window (classified finishers)."""
    res = graph_data["results"]
    mask = (
        res.in_ranking
        & (res.year >= config.min_year)
        & (res.year <= config.max_year)
        & (res.driver_state_idx >= 0)
        & (res.constructor_state_idx >= 0)
    )
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return idx

    rng = np.random.default_rng(config.seed)
    n = min(int(config.max_samples), int(idx.numel()))
    choice = rng.choice(idx.cpu().numpy(), size=n, replace=False)
    return torch.from_numpy(choice.astype(np.int64))


def _race_swap_lookup(graph_data, sample_idx: torch.Tensor) -> Dict[int, int]:
    """Map each result row index to an alternate constructor_state_idx in the same race."""
    res = graph_data["results"]
    rows = pd.DataFrame(
        {
            "idx": sample_idx.cpu().numpy(),
            "year": res.year[sample_idx].cpu().numpy(),
            "round": res.round[sample_idx].cpu().numpy(),
            "race_id": res.race_id[sample_idx].cpu().numpy(),
            "constructor_state_idx": res.constructor_state_idx[sample_idx].cpu().numpy(),
        }
    )

    full = pd.DataFrame(
        {
            "idx": np.arange(res.year.shape[0]),
            "year": res.year.cpu().numpy(),
            "round": res.round.cpu().numpy(),
            "race_id": res.race_id.cpu().numpy(),
            "constructor_state_idx": res.constructor_state_idx.cpu().numpy(),
        }
    )
    full = full[full["constructor_state_idx"] >= 0]

    lookup: Dict[int, int] = {}
    for _, row in rows.iterrows():
        same_race = full[
            (full["race_id"] == row["race_id"])
            & (full["constructor_state_idx"] != row["constructor_state_idx"])
        ]
        if same_race.empty:
            continue
        alt = int(same_race.iloc[0]["constructor_state_idx"])
        lookup[int(row["idx"])] = alt
    return lookup


@torch.no_grad()
def constructor_leakage_probe(
    model: SkillGNN,
    graph_data,
    tf_dict,
    edge_index_dict,
    device: torch.device,
    sample_idx: torch.Tensor,
) -> float:
    """Spearman rho between driver readout and constructor embedding norm per race row."""
    if sample_idx.numel() < 5:
        return float("nan")

    x_dict = model.encode(tf_dict, edge_index_dict)
    res = graph_data["results"]
    d_idx = res.driver_state_idx[sample_idx].to(device)
    c_idx = res.constructor_state_idx[sample_idx].to(device)

    d_emb = x_dict["driver_state"][d_idx]
    c_emb = x_dict["constructor_state"][c_idx]
    driver_skill = model.driver_readout(d_emb).squeeze(-1).cpu().numpy()
    constructor_norm = c_emb.norm(dim=-1).cpu().numpy()

    if np.std(driver_skill) < 1e-9 or np.std(constructor_norm) < 1e-9:
        return float("nan")
    rho, _ = spearmanr(driver_skill, constructor_norm)
    return float(rho)


@torch.no_grad()
def swap_invariance_test(
    model: SkillGNN,
    graph_data,
    tf_dict,
    edge_index_dict,
    device: torch.device,
    sample_idx: torch.Tensor,
    config: ProbeSampleConfig,
) -> Dict[str, float]:
    """Driver skill stability and utility change under constructor swap at readout."""
    if sample_idx.numel() == 0:
        return {"skill_diff": float("nan"), "utility_swap_delta": float("nan"), "n_swaps": 0}

    x_dict = model.encode(tf_dict, edge_index_dict)
    res = graph_data["results"]
    swap_lookup = _race_swap_lookup(graph_data, sample_idx)

    rng = np.random.default_rng(config.seed)
    eligible = [int(i) for i in sample_idx.cpu().numpy() if int(i) in swap_lookup]
    if not eligible:
        return {"skill_diff": float("nan"), "utility_swap_delta": float("nan"), "n_swaps": 0}

    n = min(config.swap_samples, len(eligible))
    chosen = rng.choice(eligible, size=n, replace=False)

    skill_diffs = []
    utility_deltas = []
    for row_idx in chosen:
        row_idx_cpu = torch.tensor([row_idx], dtype=torch.long)
        alt_c = torch.tensor([swap_lookup[row_idx]], device=device)
        grid = res.grid[row_idx_cpu].to(device)
        d_idx = res.driver_state_idx[row_idx_cpu].to(device)
        c_idx = res.constructor_state_idx[row_idx_cpu].to(device)

        u_orig, skill_orig = model.race_utilities(x_dict, d_idx, c_idx, grid)
        u_swap, skill_swap = model.race_utilities(x_dict, d_idx, alt_c, grid)

        skill_diffs.append(abs(float(skill_orig.item() - skill_swap.item())))
        utility_deltas.append(abs(float(u_orig.item() - u_swap.item())))

    return {
        "skill_diff": float(np.mean(skill_diffs)) if skill_diffs else float("nan"),
        "utility_swap_delta": float(np.mean(utility_deltas)) if utility_deltas else float("nan"),
        "n_swaps": len(skill_diffs),
    }


@torch.no_grad()
def channel_decomposition(
    model: SkillGNN,
    graph_data,
    tf_dict,
    edge_index_dict,
    device: torch.device,
    sample_idx: torch.Tensor,
) -> Dict[str, float]:
    """Mean absolute driver/constructor/grid contributions and driver share ratio."""
    if sample_idx.numel() == 0:
        return {
            "driver_abs_mean": float("nan"),
            "constructor_abs_mean": float("nan"),
            "grid_abs_mean": float("nan"),
            "driver_share_mean": float("nan"),
            "constructor_share_mean": float("nan"),
            "constructor_dominates": False,
        }

    x_dict = model.encode(tf_dict, edge_index_dict)
    res = graph_data["results"]
    d_idx = res.driver_state_idx[sample_idx].to(device)
    c_idx = res.constructor_state_idx[sample_idx].to(device)
    grid = res.grid[sample_idx].to(device)

    d_emb = x_dict["driver_state"][d_idx]
    c_emb = x_dict["constructor_state"][c_idx]
    u_d = model.driver_readout(d_emb).squeeze(-1)
    u_c = model.constructor_readout(c_emb).squeeze(-1)
    u_grid = model.grid_weight * (-(grid.float() - 1.0))

    abs_d = u_d.abs().cpu().numpy()
    abs_c = u_c.abs().cpu().numpy()
    abs_g = u_grid.abs().cpu().numpy()
    denom = abs_d + abs_c
    driver_share = np.where(denom > 1e-9, abs_d / denom, 0.5)

    driver_share_mean = float(np.mean(driver_share))
    return {
        "driver_abs_mean": float(np.mean(abs_d)),
        "constructor_abs_mean": float(np.mean(abs_c)),
        "grid_abs_mean": float(np.mean(abs_g)),
        "driver_share_mean": driver_share_mean,
        "constructor_share_mean": float(1.0 - driver_share_mean),
        "constructor_dominates": driver_share_mean < 0.3,
    }


def evaluate_xai_gates(leakage_rho: float, swap_skill_diff: float) -> Dict[str, bool]:
    """Apply XAI gate thresholds from skill_validation.evaluate_gates."""
    leakage_pass = abs(leakage_rho) < 0.3 if not np.isnan(leakage_rho) else False
    swap_pass = swap_skill_diff < 0.05 if not np.isnan(swap_skill_diff) else False
    return {"constructor_leakage": leakage_pass, "swap_invariance": swap_pass}


def infer_claim_level(
    partial_rho: float,
    partial_ci_low: float,
    leakage_rho: float,
) -> str:
    """Map career + XAI metrics to an allowed claim level."""
    career_pass = partial_rho >= 0.15 and partial_ci_low > 0
    leakage_pass = abs(leakage_rho) < 0.3 if not np.isnan(leakage_rho) else False

    if career_pass and leakage_pass:
        return "strong_skill"
    if career_pass:
        return "car_adjusted_performance"
    return "insufficient"


@torch.no_grad()
def run_xai_probes(
    db: Database,
    checkpoint_path: str = "output/skill_model/skill_gnn.pth",
    meta_path: str = "output/skill_model/skill_gnn_meta.json",
    config: Optional[ProbeSampleConfig] = None,
) -> Dict[str, Any]:
    """Run all SkillGNN XAI probes and return a report dict."""
    config = config or ProbeSampleConfig()
    model, graph_data, tf_dict, edge_index_dict, device = load_skill_gnn_for_probes(
        db, checkpoint_path=checkpoint_path, meta_path=meta_path
    )

    sample_idx = sample_race_rows(graph_data, config)
    leakage_rho = constructor_leakage_probe(
        model, graph_data, tf_dict, edge_index_dict, device, sample_idx
    )
    swap = swap_invariance_test(
        model, graph_data, tf_dict, edge_index_dict, device, sample_idx, config
    )
    channels = channel_decomposition(
        model, graph_data, tf_dict, edge_index_dict, device, sample_idx
    )
    gates = evaluate_xai_gates(leakage_rho, swap["skill_diff"])

    return {
        "skill_source": "skill_gnn",
        "constructor_leakage_rho": leakage_rho,
        "swap_invariance": {
            "skill_diff": swap["skill_diff"],
            "utility_swap_delta": swap["utility_swap_delta"],
            "n_swaps": swap["n_swaps"],
        },
        "channel_decomposition": channels,
        "gates": gates,
        "n_samples": int(sample_idx.numel()),
        "seed": config.seed,
    }
