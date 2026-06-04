import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
import json
import os

from sklearn.metrics import roc_auc_score

from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
import sys

sys.path.append(os.path.abspath("src"))

import config as cfg
from models.pipeline_fusion import F1OrthogonalPipeline, OrthogonalSeparationLoss, pair_cosine


# ---------------------------------------------------------------------------
# Temporal DB filtering
# ---------------------------------------------------------------------------

def filter_db_by_years(db, min_year, max_year):
    """
    Filter a RelBench database to races in [min_year, max_year] (in place).

    Reference tables (constructors, drivers, circuits, status) are kept
    unfiltered so that entity IDs remain valid across year boundaries.
    """
    races_df = db.table_dict["races"].df.copy()
    races_df = races_df[(races_df["year"] >= min_year) & (races_df["year"] <= max_year)]
    valid_race_ids = set(races_df["raceId"].unique())

    for name, table in list(db.table_dict.items()):
        if name == "races":
            table.df = races_df
        elif "raceId" in table.df.columns:
            table.df = table.df[table.df["raceId"].isin(valid_race_ids)]

    return db


# ---------------------------------------------------------------------------
# Instance building from the relational graph (no CSVs, no FastF1)
# ---------------------------------------------------------------------------

def _build_instances(db):
    results_df = db.table_dict["results"].df
    races_df = db.table_dict["races"].df

    df = results_df.merge(races_df[["raceId", "year"]], on="raceId", how="inner")
    df["top3"] = (df["positionOrder"] <= 3).astype(int)

    df = df[["driverId", "constructorId", "top3", "year", "raceId"]].copy()
    df = df.drop_duplicates(subset=["driverId", "raceId"]).reset_index(drop=True)
    return df


class F1AlignedDataset(Dataset):
    """Each sample: (driverId, constructorId, top3_label)."""

    def __init__(self, df):
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        driver_id = torch.tensor(int(row["driverId"]), dtype=torch.long)
        constructor_id = torch.tensor(int(row["constructorId"]), dtype=torch.long)
        target = torch.tensor(int(row["top3"]), dtype=torch.float32)
        return driver_id, constructor_id, target


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(db):
    stype_proposal = get_stype_proposal(db)
    for table_name, col_stypes in stype_proposal.items():
        for col_name, col_stype in col_stypes.items():
            if col_stype in (stype.text_embedded, stype.text_tokenized):
                stype_proposal[table_name][col_name] = stype.categorical
    graph_data, _ = make_pkey_fkey_graph(db, col_to_stype_dict=stype_proposal)
    return graph_data


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(min_year=2000, max_year=2023):
    print("-> Loading RelBench dataset...")
    dataset = get_dataset(cfg.RELBENCH_DATASET, download=True)
    db = dataset.get_db(upto_test_timestamp=False)

    print(f"-> Filtering DB to {min_year}-{max_year}...")
    db = filter_db_by_years(db, min_year, max_year)

    print("-> Building heterogeneous graph...")
    graph_data = build_graph(db)

    print("-> Building instances from results table...")
    df = _build_instances(db)

    train_df = df[df["year"].isin(cfg.TRAIN_YEARS)].copy()
    val_df = df[df["year"].isin(cfg.VAL_YEARS)].copy()
    test_df = df[df["year"].isin(cfg.TEST_YEARS)].copy()

    train_dataset = F1AlignedDataset(train_df)
    val_dataset = F1AlignedDataset(val_df)
    test_dataset = F1AlignedDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"-> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"   Graph nodes: driver={graph_data['driver'].num_nodes}, "
          f"constructors={graph_data['constructors'].num_nodes}")

    return train_loader, val_loader, test_loader, graph_data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _collect_latents(model, dataloader, graph_data, device):
    model.eval()
    all_vp, all_ve = [], []
    with torch.no_grad():
        for batch in dataloader:
            driver_ids, constructor_ids, _ = [b.to(device) for b in batch]
            _, _, _, vp, ve = model(
                graph_x_dict=graph_data.x_dict,
                graph_edge_index_dict=graph_data.edge_index_dict,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
            )
            all_vp.append(vp.cpu())
            all_ve.append(ve.cpu())
    vp = torch.cat(all_vp, dim=0) if all_vp else None
    ve = torch.cat(all_ve, dim=0) if all_ve else None
    return vp, ve


def evaluate(model, dataloader, graph_data, criterion, device):
    model.eval()
    epoch_loss = epoch_bce = epoch_orth = 0.0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for batch in dataloader:
            driver_ids, constructor_ids, targets = [b.to(device) for b in batch]

            logits, logits_piloto, logits_equipe, v_piloto, v_equipe = model(
                graph_x_dict=graph_data.x_dict,
                graph_edge_index_dict=graph_data.edge_index_dict,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
            )

            loss, loss_bce, loss_orth = criterion(
                logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe,
            )

            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_orth += loss_orth.item()

            preds = torch.sigmoid(logits.squeeze(-1))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    batches = len(dataloader)
    if batches == 0:
        return dict(loss=0, bce=0, orth=0, auroc=0.5)

    vp, ve = _collect_latents(model, dataloader, graph_data, device)
    cos_global = float(pair_cosine(vp, ve)) if vp is not None and ve is not None else 0.0

    try:
        auroc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auroc = 0.5

    return dict(
        loss=epoch_loss / batches,
        bce=epoch_bce / batches,
        orth=epoch_orth / batches,
        cos_global=cos_global,
        auroc=auroc,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    name,
    lambda_orthogonal,
    train_loader,
    val_loader,
    test_loader,
    graph_data,
    device,
    epochs=10,
    lr=0.001,
    aux_weight=0.5,
    latent_dim=8,
):
    print(
        f"\n--- Treinando Modelo: {name} "
        f"(lambda_orthogonal={lambda_orthogonal}, latent_dim={latent_dim}) ---"
    )

    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}
    model = F1OrthogonalPipeline(
        num_nodes_dict=num_nodes_dict,
        latent_dim=latent_dim,
    ).to(device)
    criterion = OrthogonalSeparationLoss(
        lambda_orthogonal=lambda_orthogonal,
        aux_weight=aux_weight,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_bce": [], "train_orth": [],
        "val_loss": [], "val_bce": [], "val_orth": [],
        "val_auroc": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_bce = epoch_orth = 0.0

        for batch in train_loader:
            driver_ids, constructor_ids, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()

            logits, logits_piloto, logits_equipe, v_piloto, v_equipe = model(
                graph_x_dict=graph_data.x_dict,
                graph_edge_index_dict=graph_data.edge_index_dict,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
            )

            loss, loss_bce, loss_orth = criterion(
                logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_orth += loss_orth.item()

        n = max(len(train_loader), 1)
        history["train_loss"].append(epoch_loss / n)
        history["train_bce"].append(epoch_bce / n)
        history["train_orth"].append(epoch_orth / n)

        val = evaluate(model, val_loader, graph_data, criterion, device)
        history["val_loss"].append(val["loss"])
        history["val_bce"].append(val["bce"])
        history["val_orth"].append(val["orth"])
        history["val_auroc"].append(val["auroc"])

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Val AUROC: {val['auroc']:.4f} | Val Orth: {val['orth']:.4f} | "
            f"Cos(global): {val['cos_global']:.4f}"
        )

    test = evaluate(model, test_loader, graph_data, criterion, device)
    print(
        f"Test para {name} | AUROC: {test['auroc']:.4f} | "
        f"Orth: {test['orth']:.4f} | Cos(global): {test['cos_global']:.4f}"
    )

    os.makedirs("output/models", exist_ok=True)
    model_path = f"output/models/{name}.pth"
    torch.save(model.state_dict(), model_path)

    return {
        "model_name": name,
        "configuration": {
            "lambda_orthogonal": lambda_orthogonal,
            "aux_weight": aux_weight,
            "latent_dim": latent_dim,
            "lr": lr,
            "epochs": epochs,
        },
        "history": history,
        "test_metrics": {
            "loss": test["loss"],
            "bce": test["bce"],
            "orth": test["orth"],
            "cos_global": test["cos_global"],
            "auroc": test["auroc"],
        },
        "model_path": model_path,
    }


def train_models(epochs=10, run_ablation=True):
    train_loader, val_loader, test_loader, graph_data = prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = graph_data.to(device)

    results = []

    res_orth = train_and_evaluate(
        "model_orthogonal", lambda_orthogonal=1.0,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        graph_data=graph_data, device=device, epochs=epochs,
    )
    results.append(res_orth)

    res_no_orth = train_and_evaluate(
        "model_no_orthogonal", lambda_orthogonal=0.0,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        graph_data=graph_data, device=device, epochs=epochs,
    )
    results.append(res_no_orth)

    if run_ablation:
        ablation_grid = [
            {"name": "model_ablation_l01", "lambda_orthogonal": 0.1, "aux_weight": 0.5},
            {"name": "model_ablation_l05", "lambda_orthogonal": 0.5, "aux_weight": 0.5},
            {"name": "model_ablation_l2",  "lambda_orthogonal": 2.0, "aux_weight": 0.5},
            {"name": "model_no_aux_l1",    "lambda_orthogonal": 1.0, "aux_weight": 0.0},
        ]

        for cfg_row in ablation_grid:
            res_ablation = train_and_evaluate(
                cfg_row["name"],
                lambda_orthogonal=cfg_row["lambda_orthogonal"],
                aux_weight=cfg_row["aux_weight"],
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                graph_data=graph_data, device=device, epochs=epochs,
            )
            results.append(res_ablation)

        best_model = sorted(
            results,
            key=lambda r: (-r["test_metrics"]["auroc"], r["test_metrics"]["orth"]),
        )[0]
        print(
            f"\nMelhor modelo (criterio: AUROC desc, orth asc): "
            f"{best_model['model_name']}"
        )
        print(f"Metricas: {best_model['test_metrics']}")

    os.makedirs("output/models", exist_ok=True)
    with open("output/models/training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResultados salvos em output/models/training_results.json e modelos em output/models/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train F1 Pipeline (graph-only, cosine orthogonality)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip lambda ablation models")
    args = parser.parse_args()

    train_models(
        epochs=args.epochs,
        run_ablation=not args.skip_ablation,
    )
