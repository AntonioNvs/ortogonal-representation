import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import warnings
import json
import os
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
import sys

sys.path.append(os.path.abspath("src"))

from models.pipeline_fusion import F1OrthogonalPipeline, OrthogonalSeparationLoss


def hsic_rbf(x, y, sigma_x=None, sigma_y=None):
    """
    HSIC com kernel RBF (estimativa centrada simples) para medir dependência não linear.
    Retorna 0 para batches muito pequenos.
    """
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
    hsic = torch.sum(k_center * l_center) / ((n - 1) ** 2)
    return hsic

class F1AlignedDataset(Dataset):
    def __init__(self, df_aligned):
        self.data = df_aligned.reset_index(drop=True)
        
        self.driver_features = ['avg_qualifying_pos', 'teammate_delta', 'crash_rate', 
                                'podium_rate', 'experience', 'points_per_finish']
        
        self.track_features = ['altitude_m', 'length_m', 'corners_count', 
                               'rotation', 'avg_track_temp']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        x_driver = torch.tensor(row[self.driver_features].astype(float).values, dtype=torch.float32)
        x_track = torch.tensor(row[self.track_features].astype(float).values, dtype=torch.float32)
        
        target_constructor_id = torch.tensor(int(row['constructorId']), dtype=torch.long)
        
        target = torch.tensor(int(row['top3']), dtype=torch.float32)
        
        return x_driver, x_track, target_constructor_id, target

def prepare_data_and_graph():
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
    
    df_instances = df_instances.fillna(0)
    
    df_train = df_instances[(df_instances['year'] >= 2018) & (df_instances['year'] <= 2021)]
    df_val = df_instances[df_instances['year'] == 2022]
    df_test = df_instances[df_instances['year'] == 2023]
    
    train_dataset = F1AlignedDataset(df_train)
    val_dataset = F1AlignedDataset(df_val)
    test_dataset = F1AlignedDataset(df_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"-> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    return train_loader, val_loader, test_loader, graph_data

def evaluate(model, dataloader, graph_data, criterion, device):
    model.eval()
    epoch_loss, epoch_bce, epoch_orth, epoch_crossdim, epoch_hsic = 0, 0, 0, 0, 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_driver, x_track, target_constructor_ids, targets = [b.to(device) for b in batch]
            
            logits, logits_piloto, logits_equipe, v_piloto, v_equipe = model(
                x_driver=x_driver, 
                x_track=x_track, 
                graph_x_dict=graph_data.x_dict, 
                graph_edge_index_dict=graph_data.edge_index_dict, 
                target_constructor_ids=target_constructor_ids
            )
            
            loss, loss_bce, loss_orth, loss_crossdim = criterion(
                logits, logits_piloto, logits_equipe, targets, v_piloto, v_equipe
            )
            
            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_orth += loss_orth.item()
            epoch_crossdim += loss_crossdim.item()
            epoch_hsic += hsic_rbf(v_piloto, v_equipe).item()
            
            preds = torch.sigmoid(logits.squeeze(-1))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    batches = len(dataloader)
    if batches == 0:
        return 0, 0, 0, 0, 0, 0
    
    try:
        auroc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auroc = 0.5
        
    return (
        epoch_loss / batches,
        epoch_bce / batches,
        epoch_orth / batches,
        epoch_crossdim / batches,
        epoch_hsic / batches,
        auroc,
    )

def train_and_evaluate(
    name,
    lambda_pairwise,
    lambda_crossdim,
    train_loader,
    val_loader,
    test_loader,
    graph_data,
    device,
    use_track_encoder=True,
    epochs=10,
    lr=0.001,
    aux_weight=0.5,
):
    print(
        f"\n--- Treinando Modelo: {name} "
        f"(lambda_pairwise={lambda_pairwise}, lambda_crossdim={lambda_crossdim}, "
        f"use_track_encoder={use_track_encoder}) ---"
    )
    
    num_nodes_dict = {node_type: graph_data[node_type].num_nodes for node_type in graph_data.node_types}
    model = F1OrthogonalPipeline(num_nodes_dict=num_nodes_dict, latent_dim=8, use_track_encoder=use_track_encoder).to(device)
    criterion = OrthogonalSeparationLoss(
        lambda_orthogonal=lambda_pairwise,
        lambda_crossdim=lambda_crossdim,
        aux_weight=aux_weight,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_bce': [], 'train_orth': [], 'train_crossdim': [], 'train_hsic': [],
        'val_loss': [], 'val_bce': [], 'val_orth': [], 'val_crossdim': [], 'val_hsic': [], 'val_auroc': []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_bce, epoch_orth, epoch_crossdim, epoch_hsic = 0, 0, 0, 0, 0
        
        for batch in train_loader:
            x_driver, x_track, target_constructor_ids, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            logits, logits_piloto, logits_equipe, v_piloto, v_equipe = model(
                x_driver=x_driver, 
                x_track=x_track, 
                graph_x_dict=graph_data.x_dict, 
                graph_edge_index_dict=graph_data.edge_index_dict, 
                target_constructor_ids=target_constructor_ids
            )
            
            loss, loss_bce, loss_orth, loss_crossdim = criterion(
                logits, logits_piloto, logits_equipe, targets, v_piloto, v_equipe
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_orth += loss_orth.item()
            epoch_crossdim += loss_crossdim.item()
            epoch_hsic += hsic_rbf(v_piloto.detach(), v_equipe.detach()).item()
            
        batches = len(train_loader)
        if batches > 0:
            train_l = epoch_loss / batches
            train_b = epoch_bce / batches
            train_o = epoch_orth / batches
            train_c = epoch_crossdim / batches
            train_h = epoch_hsic / batches
        else:
            train_l, train_b, train_o, train_c, train_h = 0, 0, 0, 0, 0
            
        val_l, val_b, val_o, val_c, val_h, val_a = evaluate(model, val_loader, graph_data, criterion, device)
        
        history['train_loss'].append(train_l)
        history['train_bce'].append(train_b)
        history['train_orth'].append(train_o)
        history['train_crossdim'].append(train_c)
        history['train_hsic'].append(train_h)
        history['val_loss'].append(val_l)
        history['val_bce'].append(val_b)
        history['val_orth'].append(val_o)
        history['val_crossdim'].append(val_c)
        history['val_hsic'].append(val_h)
        history['val_auroc'].append(val_a)
        
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_l:.4f} | "
            f"Val AUROC: {val_a:.4f} | Val Orth: {val_o:.4f} | "
            f"Val CrossDim: {val_c:.4f} | Val HSIC: {val_h:.4f}"
        )
        
    test_l, test_b, test_o, test_c, test_h, test_a = evaluate(model, test_loader, graph_data, criterion, device)
    print(
        f"Test para {name} | AUROC: {test_a:.4f} | Orth: {test_o:.4f} | "
        f"CrossDim: {test_c:.4f} | HSIC: {test_h:.4f}"
    )
    
    os.makedirs('output/models', exist_ok=True)
    model_path = f"output/models/{name}.pth"
    torch.save(model.state_dict(), model_path)
    
    result = {
        'model_name': name,
        'configuration': {
            'lambda_orthogonal': lambda_pairwise,
            'lambda_pairwise': lambda_pairwise,
            'lambda_crossdim': lambda_crossdim,
            'aux_weight': aux_weight,
            'latent_dim': 8,
            'lr': lr,
            'epochs': epochs,
            'use_track_encoder': use_track_encoder,
        },
        'history': history,
        'test_metrics': {
            'loss': test_l,
            'bce': test_b,
            'orth': test_o,
            'crossdim': test_c,
            'hsic': test_h,
            'auroc': test_a,
        },
        'model_path': model_path
    }
    
    return result

def train_models(use_track_encoder=True, epochs=10, run_ablation=True):
    train_loader, val_loader, test_loader, graph_data = prepare_data_and_graph()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)
    
    results = []
    
    res_orth = train_and_evaluate(
        "model_orthogonal",
        lambda_pairwise=1.0,
        lambda_crossdim=0.0,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        graph_data=graph_data,
        device=device,
        use_track_encoder=use_track_encoder,
        epochs=epochs,
    )
    results.append(res_orth)
    
    res_no_orth = train_and_evaluate(
        "model_no_orthogonal",
        lambda_pairwise=0.0,
        lambda_crossdim=0.0,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        graph_data=graph_data,
        device=device,
        use_track_encoder=use_track_encoder,
        epochs=epochs,
    )
    results.append(res_no_orth)

    if run_ablation:
        ablation_grid = [
            {"name": "model_ablation_p01_c01", "lambda_pairwise": 0.1, "lambda_crossdim": 0.1},
            {"name": "model_ablation_p1_c01", "lambda_pairwise": 1.0, "lambda_crossdim": 0.1},
            {"name": "model_ablation_p1_c1", "lambda_pairwise": 1.0, "lambda_crossdim": 1.0},
        ]

        for cfg in ablation_grid:
            res_ablation = train_and_evaluate(
                cfg["name"],
                lambda_pairwise=cfg["lambda_pairwise"],
                lambda_crossdim=cfg["lambda_crossdim"],
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                graph_data=graph_data,
                device=device,
                use_track_encoder=use_track_encoder,
                epochs=epochs,
            )
            results.append(res_ablation)

        # Seleção simples do melhor compromisso:
        # melhor AUROC; empate desempata por menor orth e menor hsic.
        best_model = sorted(
            results,
            key=lambda r: (
                -r["test_metrics"]["auroc"],
                r["test_metrics"]["orth"],
                r["test_metrics"].get("hsic", float("inf")),
            ),
        )[0]
        print(
            f"\nMelhor modelo (critério: AUROC desc, orth asc, hsic asc): "
            f"{best_model['model_name']}"
        )
        print(f"Métricas: {best_model['test_metrics']}")
    
    os.makedirs('output/models', exist_ok=True)
    with open('output/models/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nResultados salvos em output/models/training_results.json e modelos em output/models/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train F1 Pipeline')
    parser.add_argument('--exclude_track_encoder', action='store_true', help='Exclude track encoder from the pipeline')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--skip_ablation', action='store_true', help='Skip lambda ablation models')
    args = parser.parse_args()
    
    train_models(
        use_track_encoder=not args.exclude_track_encoder,
        epochs=args.epochs,
        run_ablation=not args.skip_ablation,
    )