"""Training loop for the temporal meta-node F1 graph predictor.

Trains :class:`HeteroRacePredictor` on the **beat-teammate** objective: for
each (race, team) pair, predict which of the two teammates finished ahead
using *only* the driver embeddings. The car, race, and circuit are identical
for both teammates, so the model cannot solve the task from the car — the
gradient is forced into the driver embedding, isolating driver skill from car.

This replaces the earlier position-regression objective, whose diagnostic
(diagnose_driver_signal) showed the driver embedding was dead: pairwise
accuracy ~= 0.46 on held-out seasons (chance 0.5).

Usage:
    python -m train_counterfactual --device cuda:7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config as cfg
from counterfactual.teammates import build_teammate_pairs
from data.enriched_dataset import EnrichedF1Dataset
from data.temporal_graph import build_temporal_graph
from models.hetero_race_predictor import HeteroRacePredictor


def _to_tensor(arr, dtype, device):
    return torch.tensor(np.asarray(arr), dtype=dtype, device=device)


def train_counterfactual(
    model: HeteroRacePredictor,
    data,
    static: dict[str, torch.Tensor],
    raced_in: pd.DataFrame,
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: str,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Build teammate pairs (A finished ahead of B) and split by year.
    pairs = build_teammate_pairs(raced_in)
    pair_year = pairs["year"].to_numpy()

    driver_A = _to_tensor(pairs["driver_A"], torch.long, device)
    driver_B = _to_tensor(pairs["driver_B"], torch.long, device)

    train_mask = torch.tensor(
        np.isin(pair_year, cfg.TRAIN_YEARS), dtype=torch.bool, device=device
    )
    val_mask = torch.tensor(
        np.isin(pair_year, cfg.VAL_YEARS), dtype=torch.bool, device=device
    )
    test_mask = torch.tensor(
        np.isin(pair_year, cfg.TEST_YEARS), dtype=torch.bool, device=device
    )
    print(
        f"  teammate pairs: train={int(train_mask.sum())}, "
        f"val={int(val_mask.sum())}, test={int(test_mask.sum())}"
    )

    # Label is always 1: A finished ahead of B by construction.
    labels = torch.ones(len(pairs), device=device)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        x_dict = model(data, static)
        skill_A = model.driver_skill(x_dict, driver_A)
        skill_B = model.driver_skill(x_dict, driver_B)
        logits = skill_A - skill_B
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[train_mask], labels[train_mask]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation: pairwise accuracy (chance = 0.5).
        model.eval()
        with torch.no_grad():
            x_dict = model(data, static)
            skill_A = model.driver_skill(x_dict, driver_A)
            skill_B = model.driver_skill(x_dict, driver_B)
            logits = skill_A - skill_B
            val_acc = (logits[val_mask] > 0).float().mean().item()

        history["train_loss"].append(float(loss.item()))
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={loss.item():.4f} | "
            f"val_acc={val_acc:.4f} (chance=0.50)"
        )

    # Final test metrics
    model.eval()
    with torch.no_grad():
        x_dict = model(data, static)
        skill_A = model.driver_skill(x_dict, driver_A)
        skill_B = model.driver_skill(x_dict, driver_B)
        logits = skill_A - skill_B
        test_acc = (logits[test_mask] > 0).float().mean().item()

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "hetero_race_predictor.pth")
    torch.save(model.state_dict(), model_path)

    results = {
        "config": {
            "state_dim": model.state_dim,
            "epochs": epochs,
            "lr": lr,
            "num_driver_season": model.driver_emb.num_embeddings,
            "num_constructor_season": model.constructor_emb.num_embeddings,
        },
        "objective": "beat_teammate",
        "test_acc": test_acc,
        "history": history,
    }
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nRESULTS: test_acc={test_acc:.4f} (chance=0.50)")
    print(f"Model saved to: {model_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train the temporal meta-node F1 graph predictor."
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="output/counterfactual")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print("Loading enriched F1 database...")
    db = EnrichedF1Dataset().get_db(upto_test_timestamp=False)

    print("Building temporal meta-node graph...")
    graph = build_temporal_graph(db, min_year=args.min_year or cfg.CAREER_VALIDATION_MIN_YEAR)
    print(
        f"  driver_season={graph.num_driver_seasons}, "
        f"constructor_season={graph.num_constructor_seasons}, "
        f"circuit={graph.num_circuits}, race={graph.num_races}, "
        f"raced_in={len(graph.raced_in)}"
    )

    model = HeteroRacePredictor(
        num_driver_season=graph.num_driver_seasons,
        num_constructor_season=graph.num_constructor_seasons,
        num_circuit=graph.num_circuits,
        num_race=graph.num_races,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        circuit_feat_dim=len(graph.static["circuit"][0]),
        race_feat_dim=len(graph.static["race"][0]),
    ).to(device)

    data = graph.data.to(device)
    static = {
        k: torch.tensor(v, dtype=torch.float32, device=device)
        for k, v in graph.static.items()
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    # Join driverId into raced_in (needed by build_teammate_pairs).
    raced_in = graph.raced_in.copy()
    raced_in["driverId"] = raced_in["driver_season"].map(
        graph.driver_season.set_index("node_idx")["driverId"]
    )

    train_counterfactual(
        model=model,
        data=data,
        static=static,
        raced_in=raced_in,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
