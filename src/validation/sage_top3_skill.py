"""Career-validation adapter for the original SAGE-GNN + MLP (top-3) model.

Reloads a trained :class:`F1OrthogonalPipeline` checkpoint, extracts the
driver-embedding-derived skill scalar (via the ``aux_piloto`` head), and
broadcasts it across each driver's active seasons so the standard
career-validation framework can score it.

Skill readout
-------------
The pipeline's ``aux_piloto`` head is a linear layer on ``emb(drivers)`` that
was trained *directly* against the outcome — no constructor mixed in. For a
top-3 (classification) target it outputs BCE logits; for a position/points
(regression) target it outputs an ordinal score. Either way, the value is a
scalar per driver that reflects what the encoder learned about that driver in
isolation from the car. It is the most direct answer to the question "did the
driver embedding learn any skill signal?" that this model can give.

Because the SAGE-GNN produces a *single* time-invariant embedding per driver
(no season meta-nodes), ``skill_score`` is constant across seasons for a given
driver. The framework's partial-ρ machinery still uses the season index (via
the constructor tier at T), so this is a legitimate — if coarse — skill map.

Sign convention
---------------
For position-like targets (``position`` / ``positionOrder``, lower is better),
the raw score is negated so *higher = better skill*, matching every other
scorer in the framework.

Usage:
    python -m src.experiments.career_validation --skill-source sage_top3

    # ablation checkpoint (lambda_ortho = 0):
    python -m src.experiments.career_validation --skill-source sage_top3 \\
        --sage-checkpoint output/models/model_no_orthogonal.pth
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch

import config as cfg
from counterfactual.support import compute_support
from data.enriched_dataset import EnrichedF1Dataset
from data.temporal_graph import build_temporal_graph
from models.pipeline_fusion import F1OrthogonalPipeline

DEFAULT_SAGE_CHECKPOINT = "output/models/model_orthogonal.pth"


def _load_graph_bits():
    """Rebuild the RelBench pkey/fkey graph used at training time.

    Uses ``train.build_graph`` so we get the exact same node types, edge
    types, and ``tf_dict`` the pipeline was trained on.
    """
    # Local import: train.py is at the repo root and pulls heavy deps.
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from train import build_graph, get_active_task

    task, _ = get_active_task()
    db = task.dataset.get_db(upto_test_timestamp=False)
    graph_data, node_to_col_names_dict, node_to_col_stats = build_graph(db)
    return db, graph_data, node_to_col_names_dict, node_to_col_stats, task


@torch.no_grad()
def _driver_scores_from_pipeline(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.Tensor, object]:
    """Return ``(scores_per_driver_node, db)`` where ``scores`` has shape
    ``(num_drivers,)`` and is indexed by the graph's driver node id (which
    equals ``driverId`` since RelBench builds contiguous pkeys)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAGE-GNN checkpoint not found: {checkpoint_path}. "
            "Run `python train.py` first, or pass --sage-checkpoint."
        )

    db, graph_data, node_col_names, node_col_stats, task = _load_graph_bits()

    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}

    # Match training-time latent_dim by peeking at the checkpoint. The aux
    # piloto head is a Linear(latent_dim -> 1), so its weight shape pins it.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    aux_w = ckpt.get("aux_piloto.weight")
    if aux_w is None:
        raise RuntimeError(
            f"{checkpoint_path} is not an F1OrthogonalPipeline checkpoint "
            "(no aux_piloto.weight key)."
        )
    latent_dim = int(aux_w.shape[1])

    model = F1OrthogonalPipeline(
        num_nodes_dict=num_nodes_dict,
        latent_dim=latent_dim,
        node_to_col_names_dict=node_col_names,
        node_to_col_stats=node_col_stats,
    ).to(device)

    # First forward pass initialises the lazy SAGEConv weights so the state
    # dict has something to load into. We dispatch a single-sample forward
    # through the full pipeline with dummy driver/constructor ids to trigger
    # every lazy layer, then load and re-forward for real.
    graph_data = graph_data.to(device)
    dummy_driver = torch.zeros(1, dtype=torch.long, device=device)
    dummy_constructor = torch.zeros(1, dtype=torch.long, device=device)
    dummy_qual = torch.zeros(1, dtype=torch.float32, device=device)
    dummy_grid = torch.zeros(1, dtype=torch.float32, device=device)
    _ = model(
        graph_x_dict=None,
        graph_edge_index_dict=graph_data.edge_index_dict,
        target_constructor_ids=dummy_constructor,
        target_driver_ids=dummy_driver,
        qualifying_position=dummy_qual,
        grid=dummy_grid,
        graph_tf_dict=graph_data.tf_dict,
    )

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        # `missing` on lazy modules after a warm-up forward would signal a
        # real mismatch — worth surfacing clearly.
        print(f"[sage_top3_skill] state_dict missing keys: {sorted(missing)}")
    if unexpected:
        print(f"[sage_top3_skill] ignoring unexpected keys: {sorted(unexpected)}")
    model.eval()

    # Get driver embeddings from the encoder alone (skip the classifier — we
    # don't need per-race predictions, only the per-driver skill readout).
    x_dict = model.encoder(graph_data.tf_dict) if model.encoder is not None else None
    out_dict = model.graph_encoder(x_dict, graph_data.edge_index_dict)
    driver_emb = out_dict["drivers"]  # (num_drivers, latent_dim)

    scores = model.aux_piloto(driver_emb).squeeze(-1)  # (num_drivers,)

    # Position-like targets are "lower is better" -> negate so higher = better.
    target_col = getattr(task, "target_col", None)
    if target_col in ("position", "positionOrder"):
        scores = -scores

    return scores.detach().cpu(), db


def load_sage_top3_skill(
    checkpoint_path: str = DEFAULT_SAGE_CHECKPOINT,
    device=None,
) -> pd.DataFrame:
    """Return per-(driverId, season) skill scores for the SAGE-GNN pipeline.

    Columns: ``[driverId, season, skill_score, support_score, support_bucket]``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading SAGE-GNN checkpoint: {checkpoint_path}")
    scores, db = _driver_scores_from_pipeline(checkpoint_path, device)

    # Enumerate (driverId, season) pairs from actually observed races. The
    # scorer contract is one row per active (driver, season); a driver's
    # skill_score is the (time-invariant) scalar broadcast across every
    # season they raced.
    results = db.table_dict["results"].df[["driverId", "raceId"]]
    races = db.table_dict["races"].df[["raceId", "year"]]
    active = (
        results.merge(races, on="raceId", how="left")
        .dropna(subset=["year"])
        .assign(season=lambda d: d["year"].astype(int))
        [["driverId", "season"]]
        .drop_duplicates()
        .sort_values(["driverId", "season"])
        .reset_index(drop=True)
    )
    active["driverId"] = active["driverId"].astype(int)

    n_drivers = scores.shape[0]
    valid = active[active["driverId"] < n_drivers].copy()
    dropped = len(active) - len(valid)
    if dropped:
        print(f"[sage_top3_skill] dropped {dropped} rows with driverId >= n_drivers")

    valid["skill_score"] = valid["driverId"].map(
        lambda d: float(scores[int(d)].item())
    )

    # Support score: reuse the temporal-graph support so the framework's
    # by-bucket reporting works uniformly across all scorers.
    print("Building temporal graph for support scoring...")
    graph = build_temporal_graph(EnrichedF1Dataset().get_db(upto_test_timestamp=False))
    support = compute_support(graph)

    merged = valid.merge(support, on=["driverId", "season"], how="left")
    return merged[
        ["driverId", "season", "skill_score", "support_score", "support_bucket"]
    ]
