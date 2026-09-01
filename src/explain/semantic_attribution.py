"""Semantic attribution via SkillGNN additive channel decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from explain.skill_gnn_probes import ProbeSampleConfig, channel_decomposition, sample_race_rows
from models.skill_gnn import SkillGNN


@dataclass
class SemanticAttribution:
    driver: float
    constructor: float
    context: float
    grid: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "driver": self.driver,
            "constructor": self.constructor,
            "context": self.context,
            "grid": self.grid,
            "total": self.total,
        }


@torch.no_grad()
def semantic_attribution(
    model: SkillGNN,
    graph_data,
    tf_dict,
    edge_index_dict,
    device: torch.device,
    config: Optional[ProbeSampleConfig] = None,
) -> SemanticAttribution:
    """Mean absolute channel contributions (SkillGNN has no separate context channel)."""
    config = config or ProbeSampleConfig()
    sample_idx = sample_race_rows(graph_data, config)
    channels = channel_decomposition(
        model, graph_data, tf_dict, edge_index_dict, device, sample_idx
    )
    driver = channels["driver_abs_mean"]
    constructor = channels["constructor_abs_mean"]
    grid = channels["grid_abs_mean"]
    return SemanticAttribution(
        driver=driver,
        constructor=constructor,
        context=0.0,
        grid=grid,
        total=driver + constructor + grid,
    )


def shapley_semantic_four_player(*args, **kwargs) -> Dict[str, float]:
    """Shapley decomposition is redundant for SkillGNN's additive readout."""
    raise NotImplementedError(
        "shapley_semantic_four_player is not implemented; use semantic_attribution() "
        "for SkillGNN's explicit driver/constructor/grid channels."
    )
