import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import warnings
import json
import os
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
import sys

sys.path.append(os.path.abspath("src"))

import config as cfg
from models.pipeline_fusion import F1OrthogonalPipeline, OrthogonalSeparationLoss


DRIVER_FEATURES = [
    "avg_qualifying_pos", "teammate_delta", "crash_rate",
    "podium_rate", "experience", "points_per_finish",
]
DRIVER_FEATURES_RESID = [f + "_resid" for f in DRIVER_FEATURES]
TRACK_FEATURES = [
    "altitude_m", "length_m", "corners_count",
    "rotation", "avg_track_temp",
]


def hsic_rbf(x, y, sigma_x=None, sigma_y=None):
    """HSIC com kernel RBF (estimativa centrada simples), diagnostico."""
    n = x.size(0)
    if n < 2:
        return x.new_tensor(0.0)

    dist_x = torch.cdist(x, x, p=2) ** 2
    dist_y = torch.cdist(y, y, p=2) ** 2

    eps = 1e-8
    if sigma_x is None:
        sigma_x = torch.sqrt(torch.median(dist_x.detach()) + eps)
    if sigma_y is None:
        sigma_y = torch.sqrt(torch.median(dist_y.detach()) + eps)

    sigma_x = torch.clamp(sigma_x, min=eps)
    sigma_y = torch.clamp(sigma_y, min=eps)

    k = torch.exp(-dist_x / (2.0 * sigma_x * sigma_x + eps))
    l = torch.exp(-dist_y / (2.0 * sigma_y * sigma_y + eps))

    h = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones((n, n), device=x.device)
    k_center = h @ k @ h
    l_center = h @ l @ h
    return torch.sum(k_center * l_center) / ((n - 1) ** 2)


def normalized_hsic(x, y):
    """HSIC normalizada em [0,1] = HSIC(x,y) / sqrt(HSIC(x,x)*HSIC(y,y))."""
    hxy = hsic_rbf(x, y)
    hxx = hsic_rbf(x, x)
    hyy = hsic_rbf(y, y)
    denom = torch.sqrt(torch.clamp(hxx * hyy, min=1e-12))
    return (hxy / denom).clamp(min=0.0)


class F1AlignedDataset(Dataset):
    """
    Dataset alinhado para o pipeline. Recebe DataFrame ja filtrado por split
    e ja standardizado nas colunas numericas.
    """

    def __init__(self, df_aligned, driver_features=None, track_features=None):
        self.data = df_aligned.reset_index(drop=True)
        self.driver_features = driver_features or DRIVER_FEATURES
        self.track_features = track_features or TRACK_FEATURES

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x_driver = torch.tensor(row[self.driver_features].astype(float).values, dtype=torch.float32)
        x_track = torch.tensor(row[self.track_features].astype(float).values, dtype=torch.float32)
        target_constructor_id = torch.tensor(int(row["constructorId"]), dtype=torch.long)
        target = torch.tensor(int(row["top3"]), dtype=torch.float32)

        return x_driver, x_track, target_constructor_id, target


def _split_frames(df_instances):
    """
    Aplica o split temporal centralizado em config. Se a coluna 'split'
    existir e for consistente, usa-a; caso contrario, refaz pelos anos
    declarados em config.
    """
    if "split" in df_instances.columns and set(df_instances["split"]) & {"train", "val", "test"}:
        train_df = df_instances[df_instances["split"] == "train"].copy()
        val_df = df_instances[df_instances["split"] == "val"].copy()
        test_df = df_instances[df_instances["split"] == "test"].copy()
    else:
        train_df = df_instances[df_instances["year"].isin(cfg.TRAIN_YEARS)].copy()
        val_df = df_instances[df_instances["year"].isin(cfg.VAL_YEARS)].copy()
        test_df = df_instances[df_instances["year"].isin(cfg.TEST_YEARS)].copy()

    return train_df, val_df, test_df


def _standardize(train_df, val_df, test_df, columns):
    scaler = StandardScaler()
    train_df.loc[:, columns] = scaler.fit_transform(train_df[columns].astype(float))
    if not val_df.empty:
        val_df.loc[:, columns] = scaler.transform(val_df[columns].astype(float))
    if not test_df.empty:
        test_df.loc[:, columns] = scaler.transform(test_df[columns].astype(float))
    return scaler


def prepare_data_and_graph(persist_scaler=True, use_residual_driver=False):
    print("-> Carregando Grafo do RelBench...")
    dataset = get_dataset("rel-f1", download=True)
    db = dataset.get_db(upto_test_timestamp=False)

    stype_proposal = get_stype_proposal(db)
    for table_name, col_stypes in stype_proposal.items():
        for col_name, col_stype in col_stypes.items():
            if col_stype in [stype.text_embedded, stype.text_tokenized]:
                stype_proposal[table_name][col_name] = stype.categorical

    graph_data, _ = make_pkey_fkey_graph(db, col_to_stype_dict=stype_proposal)

    x_dict = {}
    for node_type in graph_data.node_types:
        num_nodes = graph_data[node_type].num_nodes
        x_dict[node_type] = torch.ones(num_nodes, 1)
    graph_data.x_dict = x_dict

    print("-> Carregando e Alinhando CSV Tabular...")
    df_instances = pd.read_csv("output/dataset/instances.csv")

    driver_cols = DRIVER_FEATURES_RESID if use_residual_driver else DRIVER_FEATURES
    missing_cols = [c for c in df_instances.columns if c.endswith("_missing")]
    if use_residual_driver and not set(driver_cols).issubset(df_instances.columns):
        raise RuntimeError(
            "instances.csv nao possui as colunas _resid; rode o dataset_builder atualizado."
        )

    # Salvaguarda: NaN residual (eg. anos sem track) eh imputado por mediana
    # do split de treino, mantendo flags de missingness quando existirem.
    feature_cols = driver_cols + TRACK_FEATURES
    if "split" in df_instances.columns:
        train_mask = df_instances["split"] == "train"
        medians = df_instances.loc[train_mask, feature_cols].median(numeric_only=True)
    else:
        medians = df_instances[feature_cols].median(numeric_only=True)
    for col in feature_cols:
        df_instances[col] = df_instances[col].fillna(medians.get(col, 0.0))

    train_df, val_df, test_df = _split_frames(df_instances)

    scaler = _standardize(train_df, val_df, test_df, feature_cols)

    if persist_scaler:
        scaler_path = "output/dataset/feature_scaler.json"
        with open(scaler_path, "w") as f:
            json.dump({
                "feature_cols": feature_cols,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }, f, indent=2)
        print(f"   feature_scaler -> {scaler_path}")

    train_dataset = F1AlignedDataset(train_df, driver_features=driver_cols)
    val_dataset = F1AlignedDataset(val_df, driver_features=driver_cols)
    test_dataset = F1AlignedDataset(test_df, driver_features=driver_cols)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"-> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_loader, val_loader, test_loader, graph_data


def _collect_latents(model, dataloader, graph_data, device):
    """Roda o modelo em modo eval e empilha v_piloto, v_equipe e v_pista."""
    model.eval()
    all_vp, all_ve, all_vt = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x_driver, x_track, target_constructor_ids, _ = [b.to(device) for b in batch]
            _, _, _, vp, ve, vt = model(
                x_driver=x_driver,
                x_track=x_track,
                graph_x_dict=graph_data.x_dict,
                graph_edge_index_dict=graph_data.edge_index_dict,
                target_constructor_ids=target_constructor_ids,
            )
            all_vp.append(vp.cpu())
            all_ve.append(ve.cpu())
            if vt is not None:
                all_vt.append(vt.cpu())
    vp = torch.cat(all_vp, dim=0) if all_vp else None
    ve = torch.cat(all_ve, dim=0) if all_ve else None
    vt = torch.cat(all_vt, dim=0) if all_vt else None
    return vp, ve, vt


def evaluate(model, dataloader, graph_data, criterion, device):
    model.eval()
    epoch_loss = epoch_bce = epoch_orth = epoch_crossdim = epoch_hsic_loss = 0.0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for batch in dataloader:
            x_driver, x_track, target_constructor_ids, targets = [b.to(device) for b in batch]

            logits, logits_piloto, logits_equipe, v_piloto, v_equipe, v_pista = model(
                x_driver=x_driver,
                x_track=x_track,
                graph_x_dict=graph_data.x_dict,
                graph_edge_index_dict=graph_data.edge_index_dict,
                target_constructor_ids=target_constructor_ids,
            )

            loss, loss_bce, loss_orth, loss_crossdim, loss_hsic = criterion(
                logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe, v_pista,
            )

            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_orth += loss_orth.item()
            epoch_crossdim += loss_crossdim.item()
            epoch_hsic_loss += loss_hsic.item()

            preds = torch.sigmoid(logits.squeeze(-1))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    batches = len(dataloader)
    if batches == 0:
        return dict(loss=0, bce=0, orth=0, crossdim=0, hsic_loss=0, hsic_global=0, nhsic_global=0, auroc=0.5)

    # HSIC global sobre o set inteiro (uma medicao so, em vez de media por batch).
    vp, ve, vt = _collect_latents(model, dataloader, graph_data, device)
    hsic_global = float(hsic_rbf(vp, ve)) if vp is not None and ve is not None else 0.0
    nhsic_global = float(normalized_hsic(vp, ve)) if vp is not None and ve is not None else 0.0

    try:
        auroc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auroc = 0.5

    return dict(
        loss=epoch_loss / batches,
        bce=epoch_bce / batches,
        orth=epoch_orth / batches,
        crossdim=epoch_crossdim / batches,
        hsic_loss=epoch_hsic_loss / batches,
        hsic_global=hsic_global,
        nhsic_global=nhsic_global,
        auroc=auroc,
    )


def train_and_evaluate(
    name,
    lambda_pairwise,
    lambda_crossdim,
    lambda_hsic,
    train_loader,
    val_loader,
    test_loader,
    graph_data,
    device,
    use_track_encoder=True,
    epochs=10,
    lr=0.001,
    aux_weight=0.5,
    latent_dim=8,
):
    print(
        f"\n--- Treinando Modelo: {name} "
        f"(lambda_pairwise={lambda_pairwise}, lambda_crossdim={lambda_crossdim}, "
        f"lambda_hsic={lambda_hsic}, use_track_encoder={use_track_encoder}, "
        f"latent_dim={latent_dim}) ---"
    )

    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}
    model = F1OrthogonalPipeline(
        num_nodes_dict=num_nodes_dict,
        latent_dim=latent_dim,
        driver_input_dim=len(DRIVER_FEATURES),
        track_input_dim=len(TRACK_FEATURES),
        use_track_encoder=use_track_encoder,
    ).to(device)
    criterion = OrthogonalSeparationLoss(
        lambda_orthogonal=lambda_pairwise,
        lambda_crossdim=lambda_crossdim,
        lambda_hsic=lambda_hsic,
        aux_weight=aux_weight,
        include_track_pairs=use_track_encoder,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_bce": [], "train_orth": [], "train_crossdim": [], "train_hsic_loss": [],
        "val_loss": [], "val_bce": [], "val_orth": [], "val_crossdim": [], "val_hsic_loss": [],
        "val_hsic_global": [], "val_nhsic_global": [], "val_auroc": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_bce = epoch_orth = epoch_crossdim = epoch_hsic_loss = 0.0

        for batch in train_loader:
            x_driver, x_track, target_constructor_ids, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()

            logits, logits_piloto, logits_equipe, v_piloto, v_equipe, v_pista = model(
                x_driver=x_driver,
                x_track=x_track,
                graph_x_dict=graph_data.x_dict,
                graph_edge_index_dict=graph_data.edge_index_dict,
                target_constructor_ids=target_constructor_ids,
            )

            loss, loss_bce, loss_orth, loss_crossdim, loss_hsic = criterion(
                logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe, v_pista,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_orth += loss_orth.item()
            epoch_crossdim += loss_crossdim.item()
            epoch_hsic_loss += loss_hsic.item()

        n = max(len(train_loader), 1)
        history["train_loss"].append(epoch_loss / n)
        history["train_bce"].append(epoch_bce / n)
        history["train_orth"].append(epoch_orth / n)
        history["train_crossdim"].append(epoch_crossdim / n)
        history["train_hsic_loss"].append(epoch_hsic_loss / n)

        val = evaluate(model, val_loader, graph_data, criterion, device)
        history["val_loss"].append(val["loss"])
        history["val_bce"].append(val["bce"])
        history["val_orth"].append(val["orth"])
        history["val_crossdim"].append(val["crossdim"])
        history["val_hsic_loss"].append(val["hsic_loss"])
        history["val_hsic_global"].append(val["hsic_global"])
        history["val_nhsic_global"].append(val["nhsic_global"])
        history["val_auroc"].append(val["auroc"])

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Val AUROC: {val['auroc']:.4f} | Val Orth: {val['orth']:.4f} | "
            f"Val CrossDim: {val['crossdim']:.4f} | "
            f"Val HSIC(loss/batch): {val['hsic_loss']:.4f} | "
            f"Val HSIC(global): {val['hsic_global']:.4f} | "
            f"Val nHSIC: {val['nhsic_global']:.3f}"
        )

    test = evaluate(model, test_loader, graph_data, criterion, device)
    print(
        f"Test para {name} | AUROC: {test['auroc']:.4f} | Orth: {test['orth']:.4f} | "
        f"CrossDim: {test['crossdim']:.4f} | HSIC(global): {test['hsic_global']:.4f} | "
        f"nHSIC: {test['nhsic_global']:.3f}"
    )

    os.makedirs("output/models", exist_ok=True)
    model_path = f"output/models/{name}.pth"
    torch.save(model.state_dict(), model_path)

    return {
        "model_name": name,
        "configuration": {
            "lambda_orthogonal": lambda_pairwise,
            "lambda_pairwise": lambda_pairwise,
            "lambda_crossdim": lambda_crossdim,
            "lambda_hsic": lambda_hsic,
            "aux_weight": aux_weight,
            "latent_dim": latent_dim,
            "lr": lr,
            "epochs": epochs,
            "use_track_encoder": use_track_encoder,
        },
        "history": history,
        "test_metrics": {
            "loss": test["loss"],
            "bce": test["bce"],
            "orth": test["orth"],
            "crossdim": test["crossdim"],
            "hsic": test["hsic_global"],
            "nhsic": test["nhsic_global"],
            "auroc": test["auroc"],
        },
        "model_path": model_path,
    }


def train_models(use_track_encoder=True, epochs=10, run_ablation=True,
                 use_residual_driver=False):
    train_loader, val_loader, test_loader, graph_data = prepare_data_and_graph(
        use_residual_driver=use_residual_driver,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = graph_data.to(device)

    results = []

    res_orth = train_and_evaluate(
        "model_orthogonal", lambda_pairwise=1.0, lambda_crossdim=0.0, lambda_hsic=0.0,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        graph_data=graph_data, device=device, use_track_encoder=use_track_encoder, epochs=epochs,
    )
    results.append(res_orth)

    res_no_orth = train_and_evaluate(
        "model_no_orthogonal", lambda_pairwise=0.0, lambda_crossdim=0.0, lambda_hsic=0.0,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        graph_data=graph_data, device=device, use_track_encoder=use_track_encoder, epochs=epochs,
    )
    results.append(res_no_orth)

    if run_ablation:
        ablation_grid = [
            {"name": "model_ablation_p01_c01", "lambda_pairwise": 0.1, "lambda_crossdim": 0.1, "lambda_hsic": 0.0},
            {"name": "model_ablation_p1_c01",  "lambda_pairwise": 1.0, "lambda_crossdim": 0.1, "lambda_hsic": 0.0},
            {"name": "model_ablation_p1_c1",   "lambda_pairwise": 1.0, "lambda_crossdim": 1.0, "lambda_hsic": 0.0},
            {"name": "model_hsic_01",          "lambda_pairwise": 0.0, "lambda_crossdim": 0.0, "lambda_hsic": 0.1},
            {"name": "model_hsic_1",           "lambda_pairwise": 0.0, "lambda_crossdim": 0.0, "lambda_hsic": 1.0},
            {"name": "model_combo_p1_h01",     "lambda_pairwise": 1.0, "lambda_crossdim": 0.0, "lambda_hsic": 0.1},
        ]

        for cfg_row in ablation_grid:
            res_ablation = train_and_evaluate(
                cfg_row["name"],
                lambda_pairwise=cfg_row["lambda_pairwise"],
                lambda_crossdim=cfg_row["lambda_crossdim"],
                lambda_hsic=cfg_row["lambda_hsic"],
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                graph_data=graph_data, device=device,
                use_track_encoder=use_track_encoder, epochs=epochs,
            )
            results.append(res_ablation)

        best_model = sorted(
            results,
            key=lambda r: (
                -r["test_metrics"]["auroc"],
                r["test_metrics"].get("nhsic", float("inf")),
                r["test_metrics"]["orth"],
            ),
        )[0]
        print(
            f"\nMelhor modelo (criterio: AUROC desc, nHSIC asc, orth asc): "
            f"{best_model['model_name']}"
        )
        print(f"Metricas: {best_model['test_metrics']}")

    os.makedirs("output/models", exist_ok=True)
    with open("output/models/training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResultados salvos em output/models/training_results.json e modelos em output/models/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train F1 Pipeline")
    parser.add_argument("--exclude_track_encoder", action="store_true",
                        help="Exclude track encoder from the pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip lambda ablation models")
    parser.add_argument("--residual_driver", action="store_true",
                        help="Use residualized driver features (less team-contaminated)")
    args = parser.parse_args()

    train_models(
        use_track_encoder=not args.exclude_track_encoder,
        epochs=args.epochs,
        run_ablation=not args.skip_ablation,
        use_residual_driver=args.residual_driver,
    )
