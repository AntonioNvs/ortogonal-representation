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
    epoch_loss, epoch_bce, epoch_orth = 0, 0, 0
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
            
            loss, loss_bce, loss_orth = criterion(logits, logits_piloto, logits_equipe, targets, v_piloto, v_equipe)
            
            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_orth += loss_orth.item()
            
            preds = torch.sigmoid(logits.squeeze(-1))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    batches = len(dataloader)
    if batches == 0:
        return 0, 0, 0, 0
    
    try:
        auroc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auroc = 0.5
        
    return epoch_loss/batches, epoch_bce/batches, epoch_orth/batches, auroc

def train_and_evaluate(name, lambda_orth, train_loader, val_loader, test_loader, graph_data, device, use_track_encoder=True):
    print(f"\n--- Treinando Modelo: {name} (lambda={lambda_orth}, use_track_encoder={use_track_encoder}) ---")
    
    num_nodes_dict = {node_type: graph_data[node_type].num_nodes for node_type in graph_data.node_types}
    model = F1OrthogonalPipeline(num_nodes_dict=num_nodes_dict, latent_dim=8, use_track_encoder=use_track_encoder).to(device)
    criterion = OrthogonalSeparationLoss(lambda_orthogonal=lambda_orth)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    EPOCHS = 10
    
    history = {
        'train_loss': [], 'train_bce': [], 'train_orth': [],
        'val_loss': [], 'val_bce': [], 'val_orth': [], 'val_auroc': []
    }
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, epoch_bce, epoch_orth = 0, 0, 0
        
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
            
            loss, loss_bce, loss_orth = criterion(logits, logits_piloto, logits_equipe, targets, v_piloto, v_equipe)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bce += loss_bce.item()
            epoch_orth += loss_orth.item()
            
        batches = len(train_loader)
        if batches > 0:
            train_l, train_b, train_o = epoch_loss/batches, epoch_bce/batches, epoch_orth/batches
        else:
            train_l, train_b, train_o = 0, 0, 0
            
        val_l, val_b, val_o, val_a = evaluate(model, val_loader, graph_data, criterion, device)
        
        history['train_loss'].append(train_l)
        history['train_bce'].append(train_b)
        history['train_orth'].append(train_o)
        history['val_loss'].append(val_l)
        history['val_bce'].append(val_b)
        history['val_orth'].append(val_o)
        history['val_auroc'].append(val_a)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_l:.4f} | Val AUROC: {val_a:.4f}")
        
    test_l, test_b, test_o, test_a = evaluate(model, test_loader, graph_data, criterion, device)
    print(f"Test AUROC para {name}: {test_a:.4f}")
    
    os.makedirs('output/models', exist_ok=True)
    model_path = f"output/models/{name}.pth"
    torch.save(model.state_dict(), model_path)
    
    result = {
        'model_name': name,
        'configuration': {'lambda_orthogonal': lambda_orth, 'latent_dim': 8, 'lr': 0.001, 'epochs': EPOCHS},
        'history': history,
        'test_metrics': {
            'loss': test_l, 'bce': test_b, 'orth': test_o, 'auroc': test_a
        },
        'model_path': model_path
    }
    
    return result

def train_models(use_track_encoder=True):
    train_loader, val_loader, test_loader, graph_data = prepare_data_and_graph()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)
    
    results = []
    
    res_orth = train_and_evaluate("model_orthogonal", 1, train_loader, val_loader, test_loader, graph_data, device, use_track_encoder)
    results.append(res_orth)
    
    res_no_orth = train_and_evaluate("model_no_orthogonal", 0.0, train_loader, val_loader, test_loader, graph_data, device, use_track_encoder)
    results.append(res_no_orth)
    
    os.makedirs('output/models', exist_ok=True)
    with open('output/models/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nResultados salvos em output/models/training_results.json e modelos em output/models/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train F1 Pipeline')
    parser.add_argument('--exclude_track_encoder', action='store_true', help='Exclude track encoder from the pipeline')
    args = parser.parse_args()
    
    train_models(use_track_encoder=not args.exclude_track_encoder)