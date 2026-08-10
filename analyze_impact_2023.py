import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("src"))

import config as cfg
from train import prepare_data, F1AlignedDataset, get_active_task, _build_instances_from_task
from models.pipeline_fusion import F1OrthogonalPipeline

def patch_state_dict(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    patched = {}
    
    for k, v in checkpoint.items():
        if k in model_state:
            try:
                target_shape = model_state[k].shape
            except RuntimeError:
                print(f"Skipping uninitialized parameter {k}")
                continue
            
            if v.shape != target_shape:
                print(f"Patching size mismatch for {k}: {v.shape} -> {target_shape}")
                new_v = model_state[k].clone()
                slices = tuple(slice(0, min(dim_v, dim_t)) for dim_v, dim_t in zip(v.shape, target_shape))
                new_v[slices] = v[slices]
                patched[k] = new_v
            else:
                patched[k] = v
        else:
            print(f"Ignoring unexpected key: {k}")
            
    # Load patched state dict
    model.load_state_dict(patched, strict=False)
    print("Model loaded successfully with patched state_dict!")

def analyze(target_year=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Preparing data and graph...")
    # Prevent prepare_data from overwriting graph_meta.pt
    original_save = torch.save
    def safe_save(obj, f, *args, **kwargs):
        if isinstance(f, str) and "graph_meta.pt" in f:
            print(f"Prevented overwriting {f}")
            return
        return original_save(obj, f, *args, **kwargs)
    torch.save = safe_save

    loaders_and_data = prepare_data()
    (train_loader, val_loader, test_loader, graph_data,
     node_to_col_names_dict, node_to_col_stats,
     train_edge_index_dict, val_edge_index_dict, test_edge_index_dict,
     task) = loaders_and_data

    print("Loading DB for instance matching...")
    _, outcome_lookup = get_active_task()
    db = task.dataset.get_db(upto_test_timestamp=False)
    df_all = _build_instances_from_task(task, outcome_lookup)

    year = target_year if target_year is not None else int(df_all["year"].max())
    df_target = df_all[df_all["year"] == year].copy()
    if len(df_target) == 0:
        print(f"No data for {year}, falling back to {year - 1}.")
        year -= 1
        df_target = df_all[df_all["year"] == year].copy()

    eval_dataset = F1AlignedDataset(df_target)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    # Load model
    print("Initializing model...")
    num_nodes_dict = {nt: graph_data[nt].num_nodes for nt in graph_data.node_types}
    model = F1OrthogonalPipeline(
        num_nodes_dict=num_nodes_dict,
        node_to_col_names_dict=node_to_col_names_dict,
        node_to_col_stats=node_to_col_stats,
        latent_dim=32,
    )
    
    model_path = "output/models/model_orthogonal.pth"
    print("Initializing lazy parameters with dummy forward pass...")
    try:
        model.encoder(graph_data.tf_dict)
    except Exception as e:
        print("Dummy forward error:", e)
    
    patch_state_dict(model, model_path, device)
    
    model.to(device)
    model.eval()
    
    graph_data = graph_data.to(device)
    test_edge_index_dict = {et: ei.to(device) for et, ei in test_edge_index_dict.items()}
    
    # Extract names
    drivers_df = db.table_dict["drivers"].df
    constructors_df = db.table_dict["constructors"].df
    
    driver_names = drivers_df["driverRef"].to_dict() if "driverRef" in drivers_df.columns else {}
    team_names = constructors_df["constructorRef"].to_dict() if "constructorRef" in constructors_df.columns else {}
    
    results = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in eval_loader:
            driver_ids, constructor_ids, qualifying_pos, grid_pos, targets, _top3 = [b.to(device) for b in batch]
            logits, logits_piloto, logits_equipe, v_piloto, v_equipe, _ = model(
                graph_x_dict=None,
                graph_edge_index_dict=test_edge_index_dict,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
                qualifying_position=qualifying_pos,
                grid=grid_pos,
                graph_tf_dict=graph_data.tf_dict,
            )
            
            for i in range(len(driver_ids)):
                d_id = driver_ids[i].item()
                c_id = constructor_ids[i].item()
                l_p = logits_piloto[i].item()
                l_c = logits_equipe[i].item()
                t = targets[i].item()
                pred = logits[i].item()
                
                results.append({
                    "driver_id": d_id,
                    "team_id": c_id,
                    "driver_name": driver_names.get(d_id, f"Driver_{d_id}"),
                    "team_name": team_names.get(c_id, f"Team_{c_id}"),
                    "impact_driver": abs(l_p),
                    "impact_team": abs(l_c),
                    "raw_logit_driver": l_p,
                    "raw_logit_team": l_c,
                    "target": t,
                    "raw_logit_fused": pred,
                })
                
    df_res = pd.DataFrame(results)
    
    avg_driver_impact = df_res["impact_driver"].mean()
    avg_team_impact = df_res["impact_team"].mean()
    
    print(f"\n--- Overall Impact (Mean Absolute Logits) ---")
    print(f"Driver Impact: {avg_driver_impact:.4f}")
    print(f"Team Impact:   {avg_team_impact:.4f}")
    
    driver_agg = df_res.groupby(["driver_name", "team_name"]).agg({
        "impact_driver": "mean",
        "impact_team": "mean",
        "raw_logit_driver": "mean",
        "raw_logit_team": "mean",
        "target": "mean"
    }).reset_index().sort_values(by="impact_driver", ascending=False)
    
    driver_agg["impact_ratio_d_vs_t"] = driver_agg["impact_driver"] / driver_agg["impact_team"]
    
    os.makedirs("output", exist_ok=True)
    out_path = f"output/impact_analysis_{year}.csv"
    driver_agg.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature-impact analysis for a given season")
    parser.add_argument(
        "--year", type=int, default=None,
        help="Season to analyze (default: latest year available for the active split mode)",
    )
    args = parser.parse_args()
    analyze(target_year=args.year)
