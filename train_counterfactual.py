"""Training loop for the temporal meta-node F1 graph predictor.

Trains :class:`HeteroRacePredictor` to predict ``position_norm`` on each
``raced_in`` edge from the four participating node embeddings (driver,
constructor, race, circuit). The graph is small enough that we train in a
single full-batch forward pass per epoch — no mini-batching, no gradient
accumulation.

Splits are by year (from ``cfg.TRAIN_YEARS`` / ``VAL_YEARS`` / ``TEST_YEARS``).
The graph is built once over the full window and message passing runs over the
whole graph; the loss is masked to the train-year edges only. Directional
``same_driver`` / ``same_constructor`` edges (T -> T+1) keep cross-season flow
forward-in-time, so a driver's 2020 embedding never sees their 2024 results.
(Static ``circuit`` nodes aggregate races across all years — a mild transductive
leak, acceptable for v1 and noted in the design doc.)

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
from data.enriched_dataset import EnrichedF1Dataset
from data.temporal_graph import build_temporal_graph
from models.hetero_race_predictor import HeteroRacePredictor


def _to_tensor(arr, dtype, device):
    return torch.tensor(np.asarray(arr), dtype=dtype, device=device)


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def train_counterfactual(
    model: HeteroRacePredictor,
    data,
    static: dict[str, torch.Tensor],
    raced_in: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: str,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Node index tensors for the readout, in raced_in row order.
    driver_idx = _to_tensor(raced_in["driver_season"], torch.long, device)
    constructor_idx = _to_tensor(raced_in["constructor_season"], torch.long, device)
    race_idx = _to_tensor(raced_in["race"], torch.long, device)
    circuit_idx = _to_tensor(raced_in["circuit"], torch.long, device)
    y = _to_tensor(raced_in["position_norm"], torch.float32, device)

    train_mask = torch.tensor(split_masks["train"], dtype=torch.bool, device=device)
    val_mask = torch.tensor(split_masks["val"], dtype=torch.bool, device=device)
    test_mask = torch.tensor(split_masks["test"], dtype=torch.bool, device=device)

    # Constant baseline: predict the training mean everywhere.
    baseline = y[train_mask].mean().item()

    history = {"train_loss": [], "val_mae": [], "val_r2": []}

    for epoch in range(epochs):
        model.train()
        x_dict = model(data, static)
        logits = model.readout_from(
            x_dict, driver_idx, constructor_idx, race_idx, circuit_idx
        )
        loss = torch.nn.functional.mse_loss(logits[train_mask], y[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            x_dict = model(data, static)
            logits = model.readout_from(
                x_dict, driver_idx, constructor_idx, race_idx, circuit_idx
            )
            val_mae = torch.nn.functional.l1_loss(logits[val_mask], y[val_mask]).item()
            val_r2 = _r2(y[val_mask], logits[val_mask])

        history["train_loss"].append(float(loss.item()))
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={loss.item():.4f} | "
            f"val_mae={val_mae:.4f} (baseline_mae={baseline:.4f}) | "
            f"val_r2={val_r2:+.3f}"
        )

    # Final test metrics
    model.eval()
    with torch.no_grad():
        x_dict = model(data, static)
        logits = model.readout_from(x_dict, driver_idx, constructor_idx, race_idx, circuit_idx)
        test_mae = torch.nn.functional.l1_loss(logits[test_mask], y[test_mask]).item()
        test_r2 = _r2(y[test_mask], logits[test_mask])

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
        "baseline_mae": baseline,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "history": history,
    }
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nRESULTS: test_mae={test_mae:.4f} (baseline {baseline:.4f}) | test_r2={test_r2:+.3f}")
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

    # Year-based split masks over the raced_in rows.
    years = graph.raced_in["year"].to_numpy()
    train_mask = np.isin(years, cfg.TRAIN_YEARS)
    val_mask = np.isin(years, cfg.VAL_YEARS)
    test_mask = np.isin(years, cfg.TEST_YEARS)
    split_masks = {"train": train_mask, "val": val_mask, "test": test_mask}
    print(
        f"  split: train={train_mask.sum()}, val={val_mask.sum()}, "
        f"test={test_mask.sum()} raced_in rows"
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

    train_counterfactual(
        model=model,
        data=data,
        static=static,
        raced_in=graph.raced_in,
        split_masks=split_masks,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
