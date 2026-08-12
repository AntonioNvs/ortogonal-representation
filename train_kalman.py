"""
Chronological training loop for the Kalman-GNN temporal skill model.

Trains a KalmanGNNPipeline on the beat-teammate task: for each race, for each
team with 2 drivers, predict which driver finishes ahead.

The training loop is fundamentally different from the current batch-based
approach: it processes races in chronological order, maintaining a persistent
state (v_drivers, v_constructors) that evolves smoothly via Kalman updates.

Usage:
    # Smoke test (small model, 1 epoch)
    python -m train_kalman --smoke --device cuda:7

    # Full training
    python -m train_kalman --device cuda:7

    # Full training with custom config
    python -m train_kalman --epochs 10 --lr 0.0005 --device cuda:7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config as cfg
from data.kalman_dataset import (
    ChronologicalRaceList,
    SlidingWindowEdgeCache,
    build_race_batch,
)
from models.kalman_gnn import KalmanGNNPipeline
from models.kalman_losses import KalmanLossManager
from train import (
    get_device,
    load_db_and_graph,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_kalman(
    model: KalmanGNNPipeline,
    loss_manager: KalmanLossManager,
    optimizer: optim.Optimizer,
    race_list: ChronologicalRaceList,
    edge_cache: SlidingWindowEdgeCache,
    results_df,
    static_x_dict: dict,
    split_masks: dict[str, np.ndarray],
    device: torch.device,
    config: dict,
):
    """Run the chronological training loop.

    State flows continuously across the entire sequence (train+val+test).
    Gradients are only computed during training races.
    """
    train_mask = split_masks["train"]
    val_mask = split_masks.get("val", np.zeros(len(race_list), dtype=bool))
    test_mask = split_masks.get("test", np.zeros(len(race_list), dtype=bool))

    n_races = len(race_list)
    warmup = config["warmup_races"]
    accumulation_steps = config["accumulation_steps"]
    contrast_every = config.get("contrast_every", 5)

    # History tracking
    history = {
        "train_loss": [], "train_pred": [], "train_smooth": [],
        "train_contrast": [], "train_skill": [],
        "val_auroc": [], "val_accuracy": [],
    }

    for epoch in range(config["epochs"]):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'=' * 60}")

        model.train()

        # Reset state to v_0 at start of each epoch
        v_drivers, v_constructors = model.get_initial_state()
        v_drivers = v_drivers.to(device)
        v_constructors = v_constructors.to(device)

        # State history for contrastive loss
        driver_emb_history: list[torch.Tensor] = []
        driver_active_history: list[torch.Tensor] = []

        # Previous skill for consistency loss
        skill_prev = None

        # Accumulation
        epoch_losses = {"total": [], "pred": [], "smooth": [], "contrast": [], "skill": []}
        accumulation_count = 0

        for race_idx in range(n_races):
            is_train = bool(train_mask[race_idx])

            # Build batch
            batch = build_race_batch(race_idx, edge_cache, results_df, race_list)
            teammate_pairs = batch["teammate_pairs"]

            if not teammate_pairs:
                # No beat-teammate pairs → update state without loss
                with torch.no_grad():
                    edge_dict = {
                        et: ei.to(device)
                        for et, ei in batch["edge_index_dict"].items()
                    }
                    v_drivers, v_constructors = model.forward_step(
                        v_drivers, v_constructors,
                        static_x_dict, edge_dict,
                        batch["active_driver_ids"].to(device),
                        batch["active_constructor_ids"].to(device),
                    )
                continue

            # --- Forward step ---
            edge_dict = {
                et: ei.to(device)
                for et, ei in batch["edge_index_dict"].items()
            }
            active_drv = batch["active_driver_ids"].to(device)
            active_cons = batch["active_constructor_ids"].to(device)

            v_drivers_new, v_constructors_new = model.forward_step(
                v_drivers, v_constructors,
                static_x_dict, edge_dict,
                active_drv, active_cons,
            )

            # --- Compute skill ---
            skill_drv, skill_cons = model.compute_skill(v_drivers_new, v_constructors_new)

            # --- Beat-teammate prediction ---
            if teammate_pairs:
                pair_a = torch.tensor([p[0] for p in teammate_pairs], dtype=torch.long, device=device)
                pair_b = torch.tensor([p[1] for p in teammate_pairs], dtype=torch.long, device=device)
                labels = torch.tensor([p[3] for p in teammate_pairs], dtype=torch.float32, device=device)

                # Pre-race context
                qual_a = torch.tensor(
                    [batch["qualifying_positions"].get(p[0], 0.0) for p in teammate_pairs],
                    dtype=torch.float32, device=device,
                )
                qual_b = torch.tensor(
                    [batch["qualifying_positions"].get(p[1], 0.0) for p in teammate_pairs],
                    dtype=torch.float32, device=device,
                )
                grid_a = torch.tensor(
                    [batch["grids"].get(p[0], 0.0) for p in teammate_pairs],
                    dtype=torch.float32, device=device,
                )
                grid_b = torch.tensor(
                    [batch["grids"].get(p[1], 0.0) for p in teammate_pairs],
                    dtype=torch.float32, device=device,
                )

                logits = model.predict_teammate(
                    v_drivers_new, pair_a, pair_b, qual_a, qual_b, grid_a, grid_b,
                )
            else:
                logits = torch.tensor([], device=device)
                labels = torch.tensor([], device=device)

            # --- Loss computation ---
            # Store state history for contrastive loss
            if race_idx >= warmup and is_train:
                driver_emb_history.append(v_drivers_new.detach())
                active_mask = torch.zeros(v_drivers_new.shape[0], dtype=torch.bool, device=device)
                if len(active_drv) > 0:
                    active_mask[active_drv] = True
                driver_active_history.append(active_mask)

            # Compute contrastive loss periodically
            contrast_embs = None
            contrast_active = None
            if (
                race_idx >= warmup
                and is_train
                and race_idx % contrast_every == 0
                and len(driver_emb_history) >= loss_manager.contrast_gap_min + 1
            ):
                contrast_embs = driver_emb_history
                contrast_active = driver_active_history

            losses = loss_manager.compute(
                logits=logits,
                labels=labels,
                v_drivers_curr=v_drivers_new,
                v_drivers_prev=v_drivers,
                v_constructors_curr=v_constructors_new,
                v_constructors_prev=v_constructors,
                active_driver_ids=active_drv,
                active_constructor_ids=active_cons,
                driver_emb_history=contrast_embs,
                driver_active_history=contrast_active,
                skill_curr=skill_drv,
                skill_prev=skill_prev,
            )

            # Update skill_prev for next step
            skill_prev = skill_drv.detach()

            # --- Backward (only for training races) ---
            if is_train and race_idx >= warmup:
                scaled_loss = losses["total"] / accumulation_steps
                scaled_loss.backward()
                accumulation_count += 1

                if accumulation_count >= accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulation_count = 0

                for k in epoch_losses:
                    epoch_losses[k].append(losses[k].item())

            # --- Update state ---
            v_drivers = v_drivers_new.detach()
            v_constructors = v_constructors_new.detach()

            # --- Log progress ---
            if race_idx % 50 == 0 and race_idx >= warmup:
                avg_total = np.mean(epoch_losses["total"][-50:]) if epoch_losses["total"] else 0
                avg_pred = np.mean(epoch_losses["pred"][-50:]) if epoch_losses["pred"] else 0
                print(
                    f"  Race {race_idx}/{n_races} | "
                    f"total={avg_total:.4f} | pred={avg_pred:.4f} | "
                    f"teammate_pairs={len(teammate_pairs)}"
                )

        # --- End of epoch: validation ---
        val_metrics = _evaluate_split(
            model, loss_manager, race_list, edge_cache, results_df,
            static_x_dict, val_mask, device, config,
            split_name="val",
        )
        history["val_auroc"].append(val_metrics.get("auroc", 0.0))
        history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))

        # Record epoch averages
        for k in ["total", "pred", "smooth", "contrast", "skill"]:
            if epoch_losses[k]:
                history[f"train_{k}"].append(np.mean(epoch_losses[k]))
            else:
                history[f"train_{k}"].append(0.0)

        print(
            f"  Epoch {epoch + 1} summary: "
            f"train_loss={history['train_total'][-1]:.4f} | "
            f"val_auroc={history['val_auroc'][-1]:.4f} | "
            f"val_acc={history['val_accuracy'][-1]:.4f}"
        )

    # --- Test evaluation ---
    test_metrics = _evaluate_split(
        model, loss_manager, race_list, edge_cache, results_df,
        static_x_dict, test_mask, device, config,
        split_name="test",
    )

    return history, test_metrics


def _evaluate_split(
    model: KalmanGNNPipeline,
    loss_manager: KalmanLossManager,
    race_list: ChronologicalRaceList,
    edge_cache: SlidingWindowEdgeCache,
    results_df,
    static_x_dict: dict,
    split_mask: np.ndarray,
    device: torch.device,
    config: dict,
    split_name: str = "val",
) -> dict:
    """Evaluate on a specific split (val or test).

    State flows from the beginning of the sequence (using v_0) through all
    races up to the split, then predictions are collected for split races.
    """
    model.eval()
    v_drivers, v_constructors = model.get_initial_state()
    v_drivers = v_drivers.to(device)
    v_constructors = v_constructors.to(device)

    all_preds = []
    all_labels = []

    n_races = len(race_list)
    with torch.no_grad():
        for race_idx in range(n_races):
            batch = build_race_batch(race_idx, edge_cache, results_df, race_list)

            edge_dict = {
                et: ei.to(device)
                for et, ei in batch["edge_index_dict"].items()
            }
            active_drv = batch["active_driver_ids"].to(device)
            active_cons = batch["active_constructor_ids"].to(device)

            v_drivers, v_constructors = model.forward_step(
                v_drivers, v_constructors,
                static_x_dict, edge_dict,
                active_drv, active_cons,
            )

            # Collect predictions for split races
            if bool(split_mask[race_idx]) and batch["teammate_pairs"]:
                pair_a = torch.tensor(
                    [p[0] for p in batch["teammate_pairs"]], dtype=torch.long, device=device
                )
                pair_b = torch.tensor(
                    [p[1] for p in batch["teammate_pairs"]], dtype=torch.long, device=device
                )
                labels = [p[3] for p in batch["teammate_pairs"]]

                qual_a = torch.tensor(
                    [batch["qualifying_positions"].get(p[0], 0.0) for p in batch["teammate_pairs"]],
                    dtype=torch.float32, device=device,
                )
                qual_b = torch.tensor(
                    [batch["qualifying_positions"].get(p[1], 0.0) for p in batch["teammate_pairs"]],
                    dtype=torch.float32, device=device,
                )
                grid_a = torch.tensor(
                    [batch["grids"].get(p[0], 0.0) for p in batch["teammate_pairs"]],
                    dtype=torch.float32, device=device,
                )
                grid_b = torch.tensor(
                    [batch["grids"].get(p[1], 0.0) for p in batch["teammate_pairs"]],
                    dtype=torch.float32, device=device,
                )

                logits = model.predict_teammate(
                    v_drivers, pair_a, pair_b, qual_a, qual_b, grid_a, grid_b,
                )
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().tolist())
                all_labels.extend(labels)

    if not all_preds:
        print(f"  [{split_name}] No predictions collected.")
        return {"auroc": 0.5, "accuracy": 0.5, "n_pairs": 0}

    auroc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1.0 if p >= 0.5 else 0.0 for p in all_preds])

    print(f"  [{split_name}] AUROC={auroc:.4f} | Accuracy={acc:.4f} | N={len(all_preds)}")

    return {"auroc": auroc, "accuracy": acc, "n_pairs": len(all_preds)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train Kalman-GNN for F1 driver skill evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., 'cuda:7').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override KALMAN_EPOCHS.")
    parser.add_argument("--lr", type=float, default=None, help="Override KALMAN_LR.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with tiny model.")
    parser.add_argument("--output-dir", type=str, default="output/kalman", help="Output directory.")
    args = parser.parse_args()

    set_global_seed(args.seed)

    # --- Resolve config ---
    if args.smoke:
        window_size = cfg.KALMAN_SMOKE_WINDOW_SIZE
        state_dim = cfg.KALMAN_SMOKE_STATE_DIM
        msg_dim = cfg.KALMAN_SMOKE_MSG_DIM
        epochs = cfg.KALMAN_SMOKE_EPOCHS
        warmup = cfg.KALMAN_SMOKE_WARMUP
        print("=== SMOKE TEST MODE ===")
    else:
        window_size = cfg.KALMAN_WINDOW_SIZE
        state_dim = cfg.KALMAN_STATE_DIM
        msg_dim = cfg.KALMAN_MSG_DIM
        epochs = cfg.KALMAN_EPOCHS
        warmup = cfg.KALMAN_WARMUP_RACES

    epochs = args.epochs if args.epochs is not None else epochs
    lr = args.lr if args.lr is not None else cfg.KALMAN_LR

    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading database and building graph...")
    db, graph_data, node_to_col_names_dict, node_to_col_stats, instances_df, task = (
        load_db_and_graph()
    )

    results_df = db.table_dict["results"].df

    # Merge qualifying position into results for teammate extraction
    qual_df = db.table_dict["qualifying"].df[["driverId", "raceId", "position"]].rename(
        columns={"position": "qualifying_position"}
    )
    results_df = results_df.merge(qual_df, on=["driverId", "raceId"], how="left")

    # --- Build chronological race list ---
    print("Building chronological race list...")
    race_list = ChronologicalRaceList(db)
    print(f"  Total races: {len(race_list)}")

    # --- Build split masks ---
    split_masks = race_list.get_train_val_test_indices(
        cfg.TRAIN_YEARS, cfg.VAL_YEARS, cfg.TEST_YEARS
    )
    n_train = split_masks["train"].sum()
    n_val = split_masks["val"].sum()
    n_test = split_masks["test"].sum()
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test} races")

    # --- Pre-compute sliding window edge cache ---
    print(f"Pre-computing edge cache (window_size={window_size})...")
    edge_cache = SlidingWindowEdgeCache(graph_data, db, race_list, window_size=window_size)
    print(f"  Cached edges for {len(edge_cache._cache)} races")

    # --- Initialize model ---
    print(f"Initializing Kalman-GNN (state_dim={state_dim}, msg_dim={msg_dim})...")
    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}
    model = KalmanGNNPipeline(
        num_drivers=graph_data["drivers"].num_nodes,
        num_constructors=graph_data["constructors"].num_nodes,
        num_nodes_dict=num_nodes_dict,
        state_dim=state_dim,
        msg_dim=msg_dim,
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
    ).to(device)

    # Encode static node features once
    print("Encoding static node features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph_data = graph_data.to(device)
        try:
            static_x_dict = model.encode_static_features_nograd(graph_data.tf_dict)
        except Exception:
            static_x_dict = {}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # --- Initialize loss manager ---
    loss_manager = KalmanLossManager(
        lambda_pred=cfg.KALMAN_LAMBDA_PRED,
        lambda_smooth=cfg.KALMAN_LAMBDA_SMOOTH,
        lambda_contrast=cfg.KALMAN_LAMBDA_CONTRAST,
        lambda_skill=cfg.KALMAN_LAMBDA_SKILL,
        contrast_gap_min=cfg.KALMAN_CONTRAST_GAP_MIN,
        contrast_gap_max=cfg.KALMAN_CONTRAST_GAP_MAX,
        contrast_temperature=cfg.KALMAN_CONTRAST_TEMP,
    )

    # --- Initialize optimizer ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.KALMAN_WEIGHT_DECAY,
    )

    # --- Training config ---
    train_config = {
        "epochs": epochs,
        "warmup_races": warmup,
        "accumulation_steps": cfg.KALMAN_ACCUMULATION_STEPS,
        "grad_clip": cfg.KALMAN_GRAD_CLIP,
        "contrast_every": cfg.KALMAN_CONTRAST_EVERY,
    }

    print(f"\nTraining config: {train_config}")
    print(f"Loss weights: pred={cfg.KALMAN_LAMBDA_PRED}, "
          f"smooth={cfg.KALMAN_LAMBDA_SMOOTH}, "
          f"contrast={cfg.KALMAN_LAMBDA_CONTRAST}, "
          f"skill={cfg.KALMAN_LAMBDA_SKILL}")

    # --- Train ---
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print(f"{'=' * 60}")

    history, test_metrics = train_kalman(
        model=model,
        loss_manager=loss_manager,
        optimizer=optimizer,
        race_list=race_list,
        edge_cache=edge_cache,
        results_df=results_df,
        static_x_dict=static_x_dict,
        split_masks=split_masks,
        device=device,
        config=train_config,
    )

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(args.output_dir, "kalman_gnn.pth")
    torch.save(model.state_dict(), model_path)

    # Save results
    results = {
        "config": {
            "window_size": window_size,
            "state_dim": state_dim,
            "msg_dim": msg_dim,
            "epochs": epochs,
            "lr": lr,
            "smoke": args.smoke,
        },
        "history": history,
        "test_metrics": test_metrics,
    }
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=float)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test pairs: {test_metrics['n_pairs']}")
    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()