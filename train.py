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
    Filter a RelBench database to races in [min_year, max_year] (in place),
    and remap primary/foreign keys to maintain contiguous integer ranges starting at 0.
    """
    races_df = db.table_dict["races"].df.copy()
    races_df = races_df[(races_df["year"] >= min_year) & (races_df["year"] <= max_year)]
    valid_race_ids = set(races_df["raceId"].unique())

    # 1. Filter dataframes
    for name, table in list(db.table_dict.items()):
        if name == "races":
            table.df = races_df
        elif "raceId" in table.df.columns:
            table.df = table.df[table.df["raceId"].isin(valid_race_ids)].copy()

    # 2. Re-map primary keys and foreign keys for all tables to ensure they are contiguous
    # We will build mapping dictionaries for each table that has a primary key
    mappings = {}
    for name, table in db.table_dict.items():
        pkey = table.pkey_col
        if pkey is not None:
            old_keys = table.df[pkey].values
            # Create mapping from old primary key to contiguous index 0..len-1
            mapping = {old_key: i for i, old_key in enumerate(old_keys)}
            mappings[name] = mapping
            # Update the primary key in the table itself
            table.df[pkey] = np.arange(len(table.df))

    # 3. Update foreign keys in all tables based on the primary key mappings
    for name, table in db.table_dict.items():
        for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
            if pkey_table in mappings:
                mapping = mappings[pkey_table]
                table.df[fkey_col] = table.df[fkey_col].map(mapping)
                if table.df[fkey_col].isnull().any():
                    table.df = table.df[table.df[fkey_col].notnull()].copy()
                table.df[fkey_col] = table.df[fkey_col].astype(np.int64)

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

def add_edge_year_masks(db, graph_data):
    """
    For each edge type whose source table contains ``raceId``, compute
    boolean masks for train / val / test based on the race year.

    The masks are derived from the source node IDs in the edge_index:
    each source node ID corresponds to a row position in the source table,
    whose ``raceId`` maps to a year via the races table.

    Edge types that have no temporal signal (no ``raceId`` column) are
    kept fully visible in every split.
    """
    races_df = db.table_dict["races"].df
    # raceId was remapped to 0..n-1 by filter_db_by_years; build {raceId: year} dict
    year_of_race_id = dict(zip(races_df["raceId"], races_df["year"]))

    masks = {}
    for edge_type in graph_data.edge_types:
        src_table = edge_type[0]
        num_edges = graph_data[edge_type].edge_index.shape[1]

        all_true = np.ones(num_edges, dtype=bool)

        if src_table in db.table_dict:
            src_df = db.table_dict[src_table].df
            if "raceId" in src_df.columns:
                src_node_ids = graph_data[edge_type].edge_index[0].cpu().numpy()
                race_ids = src_df.iloc[src_node_ids]["raceId"].values
                years = np.array([year_of_race_id.get(rid, -1) for rid in race_ids])

                masks[edge_type] = {
                    "train": np.isin(years, cfg.TRAIN_YEARS),
                    "val": np.isin(years, list(cfg.TRAIN_YEARS) + list(cfg.VAL_YEARS)),
                    "test": all_true,
                }
                continue

        # No temporal signal → keep all edges in every split
        masks[edge_type] = {"train": all_true, "val": all_true, "test": all_true}

    return masks


# ---------------------------------------------------------------------------
# Instance building from the relational graph (no CSVs, no FastF1)
# ---------------------------------------------------------------------------

def build_graph(db):
    stype_proposal = get_stype_proposal(db)
    for table_name, col_stypes in stype_proposal.items():
        for col_name, col_stype in col_stypes.items():
            if col_stype in (stype.text_embedded, stype.text_tokenized):
                stype_proposal[table_name][col_name] = stype.categorical
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format.*")
        graph_data, col_stats_dict = make_pkey_fkey_graph(db, col_to_stype_dict=stype_proposal)

    # Extract metadata for HeteroEncoder
    node_to_col_names_dict = {}
    node_to_col_stats = {}
    for node_type in graph_data.node_types:
        tf = graph_data[node_type].tf
        node_to_col_names_dict[node_type] = tf.col_names_dict
        node_to_col_stats[node_type] = col_stats_dict[node_type]

    # Save to disk for fallback loading (e.g. in notebooks)
    os.makedirs("output/models", exist_ok=True)
    torch.save({
        "node_to_col_names_dict": node_to_col_names_dict,
        "node_to_col_stats": node_to_col_stats,
    }, "output/models/graph_meta.pt")

    return graph_data, node_to_col_names_dict, node_to_col_stats


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
    graph_data, node_to_col_names_dict, node_to_col_stats = build_graph(db)

    print("-> Computing edge year masks for temporal split...")
    masks = add_edge_year_masks(db, graph_data)
    train_edge_index_dict = {
        et: ei[:, masks[et]["train"]].contiguous()
        for et, ei in graph_data.edge_index_dict.items()
    }
    val_edge_index_dict = {
        et: ei[:, masks[et]["val"]].contiguous()
        for et, ei in graph_data.edge_index_dict.items()
    }
    test_edge_index_dict = {
        et: ei[:, masks[et]["test"]].contiguous()
        for et, ei in graph_data.edge_index_dict.items()
    }

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
    print(f"   Graph nodes: drivers={graph_data['drivers'].num_nodes}, "
          f"constructors={graph_data['constructors'].num_nodes}")
    train_total = sum(ei.shape[1] for ei in train_edge_index_dict.values())
    val_total = sum(ei.shape[1] for ei in val_edge_index_dict.values())
    test_total = sum(ei.shape[1] for ei in test_edge_index_dict.values())
    print(f"   Edges: train={train_total}, val={val_total}, test={test_total}")

    return (train_loader, val_loader, test_loader, graph_data,
            node_to_col_names_dict, node_to_col_stats,
            train_edge_index_dict, val_edge_index_dict, test_edge_index_dict)


def prepare_data_and_graph(min_year=2000, max_year=2023):
    """
    Compatibility function for notebooks. Loads data, and if a trained
    model checkpoint exists, uses its encoder to pre-populate graph_data.x_dict.
    """
    loaders_and_data = prepare_data(min_year, max_year)
    (train_loader, val_loader, test_loader, graph_data,
     node_to_col_names_dict, node_to_col_stats,
     train_edge_index_dict, val_edge_index_dict, test_edge_index_dict) = loaders_and_data

    # Attempt to load model and run encoder to pre-populate x
    meta_path = "output/models/graph_meta.pt"
    if os.path.exists(meta_path):
        try:
            model = F1OrthogonalPipeline(
                num_nodes_dict={nt: graph_data[nt].num_nodes for nt in graph_data.node_types},
                node_to_col_names_dict=node_to_col_names_dict,
                node_to_col_stats=node_to_col_stats,
            )
            model_path = "output/models/model_orthogonal.pth"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            if model.encoder is not None:
                with torch.no_grad():
                    x_dict = model.encoder(graph_data.tf_dict)
                    for node_type, x in x_dict.items():
                        graph_data[node_type].x = x
        except Exception as e:
            print(f"Warning: Failed to pre-populate graph x_dict in prepare_data_and_graph: {e}")

    return (train_loader, val_loader, test_loader, graph_data,
            train_edge_index_dict, val_edge_index_dict, test_edge_index_dict)


def hsic_rbf(X, Y, sigma=1.0):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) with an RBF kernel in PyTorch.
    """
    n = X.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=X.device)

    # Pairwise distances and RBF kernel for X
    x_norm = X.pow(2).sum(dim=1, keepdim=True)
    dist_x = x_norm + x_norm.t() - 2 * torch.mm(X, X.t())
    K = torch.exp(-dist_x / (2 * sigma**2))

    # Pairwise distances and RBF kernel for Y
    y_norm = Y.pow(2).sum(dim=1, keepdim=True)
    dist_y = y_norm + y_norm.t() - 2 * torch.mm(Y, Y.t())
    L = torch.exp(-dist_y / (2 * sigma**2))

    # Centering matrix H
    H = torch.eye(n, device=X.device) - (1.0 / n) * torch.ones((n, n), device=X.device)

    # Centered kernel matrices
    Kc = torch.mm(torch.mm(H, K), H)
    Lc = torch.mm(torch.mm(H, L), H)

    # Biased HSIC estimator
    return torch.sum(Kc * Lc) / ((n - 1) ** 2)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _collect_latents(model, dataloader, graph_data, device, edge_index_dict=None):
    model.eval()
    all_vp, all_ve = [], []
    eid = edge_index_dict if edge_index_dict is not None else graph_data.edge_index_dict
    with torch.no_grad():
        for batch in dataloader:
            driver_ids, constructor_ids, _ = [b.to(device) for b in batch]
            _, _, _, vp, ve = model(
                graph_x_dict=None,
                graph_edge_index_dict=eid,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
                graph_tf_dict=graph_data.tf_dict,
            )
            all_vp.append(vp.cpu())
            all_ve.append(ve.cpu())
    vp = torch.cat(all_vp, dim=0) if all_vp else None
    ve = torch.cat(all_ve, dim=0) if all_ve else None
    return vp, ve


def evaluate(model, dataloader, graph_data, criterion, device, edge_index_dict=None):
    model.eval()
    eid = edge_index_dict if edge_index_dict is not None else graph_data.edge_index_dict
    epoch_loss = epoch_bce = epoch_orth = 0.0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for batch in dataloader:
            driver_ids, constructor_ids, targets = [b.to(device) for b in batch]

            logits, logits_piloto, logits_equipe, v_piloto, v_equipe = model(
                graph_x_dict=None,
                graph_edge_index_dict=eid,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
                graph_tf_dict=graph_data.tf_dict,
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

    vp, ve = _collect_latents(model, dataloader, graph_data, device, edge_index_dict=eid)
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
    node_to_col_names_dict,
    node_to_col_stats,
    device,
    epochs=10,
    lr=0.001,
    aux_weight=0.5,
    latent_dim=32,
    train_edge_index_dict=None,
    val_edge_index_dict=None,
    test_edge_index_dict=None,
):
    print(
        f"\n--- Treinando Modelo: {name} "
        f"(lambda_orthogonal={lambda_orthogonal}, latent_dim={latent_dim}) ---"
    )

    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}
    model = F1OrthogonalPipeline(
        num_nodes_dict=num_nodes_dict,
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
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

            train_eid = train_edge_index_dict if train_edge_index_dict is not None else graph_data.edge_index_dict
            logits, logits_piloto, logits_equipe, v_piloto, v_equipe = model(
                graph_x_dict=None,
                graph_edge_index_dict=train_eid,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
                graph_tf_dict=graph_data.tf_dict,
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

        val_eid = val_edge_index_dict if val_edge_index_dict is not None else graph_data.edge_index_dict
        val = evaluate(model, val_loader, graph_data, criterion, device, edge_index_dict=val_eid)
        history["val_loss"].append(val["loss"])
        history["val_bce"].append(val["bce"])
        history["val_orth"].append(val["orth"])
        history["val_auroc"].append(val["auroc"])

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Val AUROC: {val['auroc']:.4f} | Val Orth: {val['orth']:.4f} | "
            f"Cos(global): {val['cos_global']:.4f}"
        )

    test_eid = test_edge_index_dict if test_edge_index_dict is not None else graph_data.edge_index_dict
    test = evaluate(model, test_loader, graph_data, criterion, device, edge_index_dict=test_eid)
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
    (train_loader, val_loader, test_loader, graph_data,
     node_to_col_names_dict, node_to_col_stats,
     train_edge_index_dict, val_edge_index_dict, test_edge_index_dict) = prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = graph_data.to(device)

    # Move split-specific edge_index_dicts to the same device
    train_edge_index_dict = {et: ei.to(device) for et, ei in train_edge_index_dict.items()}
    val_edge_index_dict = {et: ei.to(device) for et, ei in val_edge_index_dict.items()}
    test_edge_index_dict = {et: ei.to(device) for et, ei in test_edge_index_dict.items()}

    results = []

    res_orth = train_and_evaluate(
        "model_orthogonal", lambda_orthogonal=1.0,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        graph_data=graph_data,
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
        device=device, epochs=epochs,
        train_edge_index_dict=train_edge_index_dict,
        val_edge_index_dict=val_edge_index_dict,
        test_edge_index_dict=test_edge_index_dict,
    )
    results.append(res_orth)

    res_no_orth = train_and_evaluate(
        "model_no_orthogonal", lambda_orthogonal=0.0,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        graph_data=graph_data,
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
        device=device, epochs=epochs,
        train_edge_index_dict=train_edge_index_dict,
        val_edge_index_dict=val_edge_index_dict,
        test_edge_index_dict=test_edge_index_dict,
    )
    results.append(res_no_orth)

    if run_ablation:
        ablation_grid = [
            {"name": "model_ablation_l01", "lambda_orthogonal": 0.1, "aux_weight": 0.5},
            #{"name": "model_ablation_l05", "lambda_orthogonal": 0.5, "aux_weight": 0.5},
            #{"name": "model_ablation_l2",  "lambda_orthogonal": 2.0, "aux_weight": 0.5},
            #{"name": "model_no_aux_l1",    "lambda_orthogonal": 1.0, "aux_weight": 0.0},
        ]

        for cfg_row in ablation_grid:
            res_ablation = train_and_evaluate(
                cfg_row["name"],
                lambda_orthogonal=cfg_row["lambda_orthogonal"],
                aux_weight=cfg_row["aux_weight"],
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                graph_data=graph_data,
                node_to_col_names_dict=node_to_col_names_dict,
                node_to_col_stats=node_to_col_stats,
                device=device, epochs=epochs,
                train_edge_index_dict=train_edge_index_dict,
                val_edge_index_dict=val_edge_index_dict,
                test_edge_index_dict=test_edge_index_dict,
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
