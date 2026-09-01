"""History attribution via relation-group ablation and integrated gradients."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from explain.skill_gnn_probes import (
    ProbeSampleConfig,
    constructor_leakage_probe as _constructor_leakage_probe,
    sample_race_rows,
    swap_invariance_test as _swap_invariance_test,
)
from models.skill_gnn import SkillGNN


def integrated_gradients_driver_emb(*args, **kwargs):
    raise NotImplementedError(
        "integrated_gradients_driver_emb is not yet implemented for SkillGNN."
    )


@torch.no_grad()
def constructor_leakage_probe(
    model: SkillGNN,
    graph_data,
    tf_dict,
    edge_index_dict,
    device: torch.device,
    config: Optional[ProbeSampleConfig] = None,
) -> float:
    config = config or ProbeSampleConfig()
    sample_idx = sample_race_rows(graph_data, config)
    return _constructor_leakage_probe(
        model, graph_data, tf_dict, edge_index_dict, device, sample_idx
    )


@torch.no_grad()
def swap_invariance_test(
    model: SkillGNN,
    graph_data,
    tf_dict,
    edge_index_dict,
    device: torch.device,
    config: Optional[ProbeSampleConfig] = None,
) -> Dict[str, float]:
    config = config or ProbeSampleConfig()
    sample_idx = sample_race_rows(graph_data, config)
    return _swap_invariance_test(
        model, graph_data, tf_dict, edge_index_dict, device, sample_idx, config
    )


def history_ablation_delta(*args, **kwargs) -> List[Dict]:
    raise NotImplementedError(
        "history_ablation_delta is not yet implemented for SkillGNN."
    )
