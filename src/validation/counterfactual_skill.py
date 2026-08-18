"""Career-validation adapter for the counterfactual driver-in-car swap.

Exposes the standard "skill scorer" contract — a function returning
``[driverId, season, skill_score, support_score, support_bucket]`` — so the
counterfactual skill plugs into ``career_validation.py`` unchanged:

    python -m src.experiments.career_validation \\
        --skill-source counterfactual_swap --require-full-horizon

The adapter rebuilds the temporal graph, replays the trained
:class:`HeteroRacePredictor`, and computes each driver's counterfactual skill
(expected outcome in an average car) plus its identification support.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch

from counterfactual.support import compute_support
from counterfactual.swap import compute_counterfactual_skill
from data.enriched_dataset import EnrichedF1Dataset
from data.temporal_graph import build_temporal_graph
from models.hetero_race_predictor import HeteroRacePredictor

DEFAULT_CHECKPOINT = "output/counterfactual/hetero_race_predictor.pth"


def load_counterfactual_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device=None,
) -> tuple[HeteroRacePredictor, object, dict[str, torch.Tensor], torch.device]:
    """Load the trained predictor, the temporal graph, and the refined
    node-embedding dict, ready for inference.

    Returns ``(model, graph, x_dict, device)``. ``x_dict`` is the output of
    ``model(data, static)`` (already on ``device``). Reused by the swap
    scorer and the driver-signal diagnostics so the checkpoint-loading
    logic lives in one place.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Counterfactual checkpoint not found: {checkpoint_path}. "
            "Run `python -m train_counterfactual` first."
        )

    print("Loading enriched F1 database...")
    db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)

    print("Building temporal meta-node graph...")
    graph = build_temporal_graph(db)

    # Reconstruct the model with the same dims used at training time. We read
    # the embedding-table sizes from the checkpoint, not from config, so a
    # checkpoint is self-describing.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    w_driver = ckpt["driver_emb.weight"]
    w_constructor = ckpt["constructor_emb.weight"]

    model = HeteroRacePredictor(
        num_driver_season=w_driver.shape[0],
        num_constructor_season=w_constructor.shape[0],
        num_circuit=graph.num_circuits,
        num_race=graph.num_races,
        state_dim=w_driver.shape[1],
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint missing parameters: {sorted(missing)}")
    if unexpected:
        print(f"[counterfactual_skill] ignoring unexpected keys: {sorted(unexpected)}")
    model.eval()

    data = graph.data.to(device)
    static = {
        k: torch.tensor(v, dtype=torch.float32, device=device)
        for k, v in graph.static.items()
    }

    with torch.no_grad():
        x_dict = model(data, static)

    return model, graph, x_dict, device


def load_counterfactual_swap_skill(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device=None,
) -> pd.DataFrame:
    """Replay the trained predictor and return counterfactual skill scores.

    Returns columns: ``[driverId, season, skill_score, support_score,
    support_bucket]``.
    """
    model, graph, x_dict, device = load_counterfactual_model(checkpoint_path, device)
    skill = compute_counterfactual_skill(model, graph, x_dict, device)
    support = compute_support(graph)

    merged = skill.merge(support, on=["driverId", "season"], how="left")
    return merged[["driverId", "season", "skill_score", "support_score", "support_bucket"]]
