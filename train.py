import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import warnings
import json
import os
import random
import argparse

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.metrics import mae as relbench_mae, rmse as relbench_rmse, r2 as relbench_r2
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
import sys

sys.path.append(os.path.abspath("src"))

import config as cfg
import data.tasks as data_tasks
from models.pipeline_fusion import F1OrthogonalPipeline, \
    OrthogonalSeparationLoss, \
    pair_cosine, \
    ORTH_MODE_PAIRED_DRIVER_CONSTRUCTOR, \
    TASK_KIND_REGRESSION

DEFAULT_MODEL_CONFIGS = {
    "zero": {"name": "model_no_orthogonal", "lambda_orthogonal": 0.0, "aux_weight": 0.5},
    "low": {"name": "model_ablation_l01", "lambda_orthogonal": 0.1, "aux_weight": 0.5},
    "high": {"name": "model_orthogonal", "lambda_orthogonal": 1.0, "aux_weight": 0.5},
}

# Columns that identify a driver's race outcome. Kept for the auxiliary
# ``auroc_top3`` metric (a top-3 finish is still a meaningful, easy-to-read
# signal for the driver-vs-constructor ranking application even though the
# primary target is now a regression on position/points), independent of
# whichever column is the active regression target.
OUTCOME_LOOKUP_COLUMNS = ["position", "positionOrder", "points", "statusId"]


def set_global_seed(seed, deterministic=True):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id=None):
    """Return the torch device, defaulting to cfg.DEFAULT_GPU_ID."""
    if gpu_id is None:
        gpu_id = cfg.DEFAULT_GPU_ID
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU {gpu_id} is unavailable. "
                f"Visible devices: {torch.cuda.device_count()}"
            )
        device = torch.device(f"cuda:{gpu_id}")
        print(f"-> Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        return device
    print("-> CUDA unavailable, using CPU")
    return torch.device("cpu")


def parse_model_grid(model_grid):
    if isinstance(model_grid, (list, tuple)):
        keys = [str(item).strip().lower() for item in model_grid if str(item).strip()]
    else:
        keys = [item.strip().lower() for item in str(model_grid).split(",") if item.strip()]

    if not keys:
        raise ValueError("Model grid cannot be empty.")

    invalid = sorted(set(keys) - set(DEFAULT_MODEL_CONFIGS.keys()))
    if invalid:
        raise ValueError(
            f"Unknown model grid entries: {invalid}. "
            f"Valid options: {sorted(DEFAULT_MODEL_CONFIGS.keys())}."
        )

    # Keep input order while de-duplicating.
    unique_keys = list(dict.fromkeys(keys))
    return [dict(DEFAULT_MODEL_CONFIGS[key], model_level=key) for key in unique_keys]


# ---------------------------------------------------------------------------
# Task / dataset resolution (RelBench AutoCompleteTask-based)
# ---------------------------------------------------------------------------

def setup_registries():
    """Registers the enriched dataset + all task variants for both split
    modes, forwarding the active temporal window from ``cfg`` so
    ``src/config.py`` remains the single source of truth. Idempotent: safe
    to call multiple times per process (``get_dataset``/``get_task`` are
    themselves ``lru_cache``-d by name, so re-registration only replaces the
    registry entry, it does not rebuild an already-instantiated dataset)."""
    data_tasks.register_all(
        enriched_db_dir=cfg.ENRICHED_DB_DIR,
        min_year=cfg.MIN_YEAR,
        max_year=cfg.MAX_YEAR,
        val_timestamp=cfg.EXTENDED_VAL_TIMESTAMP,
        test_timestamp=cfg.EXTENDED_TEST_TIMESTAMP,
    )


def get_active_task():
    """Instantiate the task selected by ``cfg`` (``TASK_NAME``,
    ``SPLIT_MODE``) and a leakage-free snapshot of raw outcome columns
    (``position``, ``positionOrder``, ``points``, ``statusId``) keyed by
    ``resultId``, captured *before* task instantiation removes them from the
    live database. The snapshot is only used for the auxiliary
    ``auroc_top3`` evaluation metric -- it never touches the graph/model
    inputs, so it cannot leak into training.

    Returns (task, outcome_lookup_df).
    """
    setup_registries()
    dataset_name = cfg.active_dataset_name()
    downloadable = dataset_name == cfg.RELBENCH_DATASET

    raw_dataset = get_dataset(dataset_name, download=downloadable)
    raw_db = raw_dataset.get_db(upto_test_timestamp=False)
    results_df = raw_db.table_dict["results"].df
    outcome_cols = [c for c in OUTCOME_LOOKUP_COLUMNS if c in results_df.columns]
    outcome_lookup = results_df[["resultId"] + outcome_cols].copy()

    task = get_task(dataset_name, cfg.TASK_NAME, download=downloadable)
    return task, outcome_lookup


# ---------------------------------------------------------------------------
# Instance building from the task's own split tables
# ---------------------------------------------------------------------------

def _build_instances_from_task(task, outcome_lookup):
    """Builds a single DataFrame covering the task's train/val/test splits,
    joined against the (leakage-free) results/races tables for identifiers
    and temporal metadata, plus an auxiliary ``top3`` label reconstructed
    from ``outcome_lookup`` (never fed to the model).

    Also joins ``qualifying.position`` (qualifying grid) as the primary
    pre-race input signal and ``results.grid`` (actual starting grid) as
    an auxiliary pre-race feature.

    Columns: driverId, constructorId, y (target), top3, year, round,
    raceId, resultId, split, qualifying_position, grid.
    """
    db = task.dataset.get_db(upto_test_timestamp=False)
    results_lookup = db.table_dict["results"].df[["resultId", "raceId", "driverId", "constructorId", "grid"]]
    races_lookup = db.table_dict["races"].df[["raceId", "year", "round"]]

    # Join qualifying position (pre-race signal) on (driverId, raceId)
    qual_lookup = db.table_dict["qualifying"].df[["driverId", "raceId", "position"]].rename(
        columns={"position": "qualifying_position"}
    )

    frames = []
    for split in ("train", "val", "test"):
        split_df = task.get_table(split, mask_input_cols=False).df
        merged = split_df.merge(results_lookup, on=task.entity_col, how="inner")
        merged = merged.merge(races_lookup, on="raceId", how="inner")
        merged = merged.merge(qual_lookup, on=["driverId", "raceId"], how="left")
        merged = merged.rename(columns={task.target_col: "y"})
        merged["split"] = split
        frames.append(merged)

    combined = pd.concat(frames, ignore_index=True)

    if "positionOrder" in outcome_lookup.columns:
        combined = combined.merge(
            outcome_lookup[["resultId", "positionOrder"]], on="resultId", how="left"
        )
        combined["top3"] = (combined["positionOrder"] <= 3).astype(int)
        combined = combined.drop(columns=["positionOrder"])
    else:
        combined["top3"] = 0

    combined = combined[
        ["driverId", "constructorId", "y", "top3", "year", "round", "raceId", "resultId",
         "split", "qualifying_position", "grid"]
    ].dropna(subset=["y"])
    combined = combined.drop_duplicates(subset=["driverId", "raceId"]).reset_index(drop=True)
    return combined


def get_race_metadata(db):
    """Return raceId -> (year, round) and year -> sorted list of rounds."""
    races_df = db.table_dict["races"].df
    race_id_to_meta = {
        int(rid): (int(year), int(rnd))
        for rid, year, rnd in zip(races_df["raceId"], races_df["year"], races_df["round"])
    }
    rounds_by_year = {
        int(year): sorted(group["round"].astype(int).unique().tolist())
        for year, group in races_df.groupby("year")
    }
    return race_id_to_meta, rounds_by_year


def filter_instances(df, years=None, target_year=None, max_round=None, exact_round=None):
    """Filter prediction instances by year and/or round within a target season."""
    if years is not None and target_year is None:
        return df[df["year"].isin(years)].copy()

    if target_year is not None:
        mask = df["year"] == target_year
        if max_round is not None:
            mask &= df["round"] <= max_round
        if exact_round is not None:
            mask &= df["round"] == exact_round
        return df[mask].copy()

    return df.copy()


def filter_curve_train_instances(df, target_year, k):
    """Historical train years plus target-season rounds 1..k."""
    historical = filter_instances(df, years=cfg.TRAIN_YEARS)
    season_partial = filter_instances(df, target_year=target_year, max_round=k)
    return (
        pd.concat([historical, season_partial], ignore_index=True)
        .drop_duplicates(subset=["driverId", "raceId"])
        .reset_index(drop=True)
    )


class F1AlignedDataset(Dataset):
    """Each sample: (driverId, constructorId, qualifying_position, grid, y, top3).

    ``qualifying_position`` is the primary pre-race input signal (qualifying
    grid). ``grid`` is the actual starting grid position (may differ from
    qualifying due to penalties). Both are pre-race features.

    ``y`` is the active regression target (position / positionOrder /
    points, per ``cfg.TASK_NAME``); ``top3`` is an auxiliary binary label
    (never used as model input) kept only so ``evaluate()`` can report a
    ranking-flavoured ``auroc_top3`` alongside the primary regression
    metrics.
    """

    def __init__(self, df):
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        driver_id = torch.tensor(int(row["driverId"]), dtype=torch.long)
        constructor_id = torch.tensor(int(row["constructorId"]), dtype=torch.long)
        qualifying_position = torch.tensor(float(row["qualifying_position"]) if pd.notna(row["qualifying_position"]) else 0.0, dtype=torch.float32)
        grid = torch.tensor(float(row["grid"]) if pd.notna(row["grid"]) else 0.0, dtype=torch.float32)
        target = torch.tensor(float(row["y"]), dtype=torch.float32)
        top3 = torch.tensor(float(row["top3"]), dtype=torch.float32)
        return driver_id, constructor_id, qualifying_position, grid, target, top3


# ---------------------------------------------------------------------------
# Graph construction / temporal masks
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


def add_edge_round_masks(db, graph_data, target_year, max_round):
    """
    For each edge type whose source table contains ``raceId``, compute a boolean
    mask that keeps edges from years < target_year or from target_year rounds
    1..max_round. Prevents leakage of future races in the target season.
    """
    races_df = db.table_dict["races"].df
    year_of_race_id = dict(zip(races_df["raceId"], races_df["year"]))
    round_of_race_id = dict(zip(races_df["raceId"], races_df["round"]))

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
                visible = []
                for rid in race_ids:
                    year = year_of_race_id.get(rid, -1)
                    rnd = round_of_race_id.get(rid, -1)
                    if year < target_year or (year == target_year and rnd <= max_round):
                        visible.append(True)
                    else:
                        visible.append(False)
                masks[edge_type] = np.array(visible, dtype=bool)
                continue

        masks[edge_type] = all_true

    return masks


def edge_index_dict_from_masks(graph_data, masks):
    return {
        et: ei[:, masks[et]].contiguous()
        for et, ei in graph_data.edge_index_dict.items()
    }


def prepare_curve_step(db, graph_data, instances_df, target_year, k, batch_size=64):
    """
    Build loaders and edge masks for walk-forward step k:
    train on historical years + target season rounds 1..k, evaluate round k+1.
    """
    train_df = filter_curve_train_instances(instances_df, target_year, k)
    val_df = filter_instances(instances_df, years=cfg.VAL_YEARS)
    eval_df = filter_instances(instances_df, target_year=target_year, exact_round=k + 1)

    masks = add_edge_round_masks(db, graph_data, target_year, k)
    edge_index_dict = edge_index_dict_from_masks(graph_data, masks)

    train_loader = DataLoader(F1AlignedDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(F1AlignedDataset(val_df), batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(F1AlignedDataset(eval_df), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, eval_loader, edge_index_dict, eval_df


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


def load_db_and_graph():
    """Load the active task's (leakage-free) DB and build the graph, without
    task-split loaders. Used by the walk-forward temporal curve experiment,
    which builds its own per-race splits directly from ``instances_df``
    rather than from the task's train/val/test tables."""
    print("-> Resolving task and dataset from config...")
    task, outcome_lookup = get_active_task()
    db = task.dataset.get_db(upto_test_timestamp=False)

    print("-> Building heterogeneous graph...")
    graph_data, node_to_col_names_dict, node_to_col_stats = build_graph(db)
    print("-> Building instances from task splits...")
    instances_df = _build_instances_from_task(task, outcome_lookup)

    return db, graph_data, node_to_col_names_dict, node_to_col_stats, instances_df, task


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data():
    print("-> Resolving task and dataset from config...")
    task, outcome_lookup = get_active_task()
    db = task.dataset.get_db(upto_test_timestamp=False)

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

    print(f"-> Building instances from task splits (task={task.__class__.__name__}, target={task.target_col})...")
    df = _build_instances_from_task(task, outcome_lookup)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

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
            train_edge_index_dict, val_edge_index_dict, test_edge_index_dict,
            task)


def prepare_data_and_graph():
    """
    Compatibility function for notebooks. Loads data, and if a trained
    model checkpoint exists, uses its encoder to pre-populate graph_data.x_dict.
    """
    loaders_and_data = prepare_data()
    (train_loader, val_loader, test_loader, graph_data,
     node_to_col_names_dict, node_to_col_stats,
     train_edge_index_dict, val_edge_index_dict, test_edge_index_dict,
     task) = loaders_and_data

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
            train_edge_index_dict, val_edge_index_dict, test_edge_index_dict,
            task)


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
            driver_ids, constructor_ids, qualifying_pos, grid_pos, _, _ = [b.to(device) for b in batch]
            _, _, _, vp, ve, _ = model(
                graph_x_dict=None,
                graph_edge_index_dict=eid,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
                qualifying_position=qualifying_pos,
                grid=grid_pos,
                graph_tf_dict=graph_data.tf_dict,
            )
            all_vp.append(vp.cpu())
            all_ve.append(ve.cpu())
    vp = torch.cat(all_vp, dim=0) if all_vp else None
    ve = torch.cat(all_ve, dim=0) if all_ve else None
    return vp, ve


def evaluate(model, dataloader, graph_data, criterion, device, edge_index_dict=None, task=None):
    """Runs the model over ``dataloader`` and reports the official RelBench
    regression metrics (r2/mae/rmse, via ``relbench.metrics``) plus
    Spearman rank correlation and an auxiliary ``auroc_top3`` (whether the
    model's raw prediction correctly ranks the actual top-3 finishers --
    inverted for position-like targets, where *lower* is better)."""
    model.eval()
    eid = edge_index_dict if edge_index_dict is not None else graph_data.edge_index_dict
    epoch_loss = epoch_main = epoch_orth = 0.0
    all_targets, all_preds, all_top3 = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            driver_ids, constructor_ids, qualifying_pos, grid_pos, targets, top3 = [b.to(device) for b in batch]

            (
                logits,
                logits_piloto,
                logits_equipe,
                v_piloto,
                v_equipe,
                paired_orthogonal_loss,
            ) = model(
                graph_x_dict=None,
                graph_edge_index_dict=eid,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
                qualifying_position=qualifying_pos,
                grid=grid_pos,
                graph_tf_dict=graph_data.tf_dict,
            )

            loss, loss_main, loss_orth = criterion(
                logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe, paired_orthogonal_loss,
            )

            epoch_loss += loss.item()
            epoch_main += loss_main.item()
            epoch_orth += loss_orth.item()

            preds = logits.squeeze(-1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_top3.extend(top3.cpu().numpy())

    batches = len(dataloader)
    if batches == 0:
        return dict(loss=0, main=0, orth=0, mae=0, rmse=0, r2=0, spearman=0, auroc_top3=0.5, cos_global=0)

    vp, ve = _collect_latents(model, dataloader, graph_data, device, edge_index_dict=eid)
    cos_global = float(pair_cosine(vp, ve)) if vp is not None and ve is not None else 0.0

    targets_arr = np.array(all_targets, dtype=np.float64)
    preds_arr = np.array(all_preds, dtype=np.float64)

    mae_val = float(relbench_mae(targets_arr, preds_arr))
    rmse_val = float(relbench_rmse(targets_arr, preds_arr))
    r2_val = float(relbench_r2(targets_arr, preds_arr))

    try:
        spearman_corr, _ = spearmanr(targets_arr, preds_arr)
        spearman_val = float(spearman_corr) if spearman_corr == spearman_corr else 0.0
    except Exception:
        spearman_val = 0.0

    # position/positionOrder are "lower is better" -- a driver predicted to
    # finish closer to 1st should score *higher* for the top3 classifier.
    lower_is_better = task is not None and getattr(task, "target_col", None) in ("position", "positionOrder")
    score_direction = -1.0 if lower_is_better else 1.0

    top3_arr = np.array(all_top3)
    try:
        auroc_top3 = roc_auc_score(top3_arr, score_direction * preds_arr)
    except ValueError:
        auroc_top3 = 0.5

    return dict(
        loss=epoch_loss / batches,
        main=epoch_main / batches,
        orth=epoch_orth / batches,
        cos_global=cos_global,
        mae=mae_val,
        rmse=rmse_val,
        r2=r2_val,
        spearman=spearman_val,
        auroc_top3=float(auroc_top3),
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
    save_model=True,
    task=None,
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
        mode=ORTH_MODE_PAIRED_DRIVER_CONSTRUCTOR,
        task=TASK_KIND_REGRESSION,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_main": [], "train_orth": [],
        "val_loss": [], "val_main": [], "val_orth": [],
        "val_mae": [], "val_rmse": [], "val_r2": [], "val_spearman": [], "val_auroc_top3": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_main = epoch_orth = 0.0

        for batch in train_loader:
            driver_ids, constructor_ids, qualifying_pos, grid_pos, targets, _top3 = [b.to(device) for b in batch]
            optimizer.zero_grad()

            train_eid = train_edge_index_dict if train_edge_index_dict is not None else graph_data.edge_index_dict
            (
                logits,
                logits_piloto,
                logits_equipe,
                v_piloto,
                v_equipe,
                paired_orthogonal_loss,
            ) = model(
                graph_x_dict=None,
                graph_edge_index_dict=train_eid,
                target_constructor_ids=constructor_ids,
                target_driver_ids=driver_ids,
                qualifying_position=qualifying_pos,
                grid=grid_pos,
                graph_tf_dict=graph_data.tf_dict,
            )

            loss, loss_main, loss_orth = criterion(
                logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe, paired_orthogonal_loss,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_main += loss_main.item()
            epoch_orth += loss_orth.item()

        n = max(len(train_loader), 1)
        history["train_loss"].append(epoch_loss / n)
        history["train_main"].append(epoch_main / n)
        history["train_orth"].append(epoch_orth / n)

        val_eid = val_edge_index_dict if val_edge_index_dict is not None else graph_data.edge_index_dict
        val = evaluate(model, val_loader, graph_data, criterion, device, edge_index_dict=val_eid, task=task)
        history["val_loss"].append(val["loss"])
        history["val_main"].append(val["main"])
        history["val_orth"].append(val["orth"])
        history["val_mae"].append(val["mae"])
        history["val_rmse"].append(val["rmse"])
        history["val_r2"].append(val["r2"])
        history["val_spearman"].append(val["spearman"])
        history["val_auroc_top3"].append(val["auroc_top3"])

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Val MAE: {val['mae']:.4f} | Val Spearman: {val['spearman']:.4f} | "
            f"Val AUROC(top3): {val['auroc_top3']:.4f} | Val Orth: {val['orth']:.4f} | "
            f"Cos(global): {val['cos_global']:.4f}"
        )

    test_eid = test_edge_index_dict if test_edge_index_dict is not None else graph_data.edge_index_dict
    test = evaluate(model, test_loader, graph_data, criterion, device, edge_index_dict=test_eid, task=task)
    print(
        f"Test para {name} | MAE: {test['mae']:.4f} | RMSE: {test['rmse']:.4f} | "
        f"R2: {test['r2']:.4f} | Spearman: {test['spearman']:.4f} | "
        f"AUROC(top3): {test['auroc_top3']:.4f} | "
        f"Orth: {test['orth']:.4f} | Cos(global): {test['cos_global']:.4f}"
    )

    os.makedirs("output/models", exist_ok=True)
    model_path = f"output/models/{name}.pth"
    if save_model:
        torch.save(model.state_dict(), model_path)
    else:
        model_path = None

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
            "main": test["main"],
            "orth": test["orth"],
            "cos_global": test["cos_global"],
            "mae": test["mae"],
            "rmse": test["rmse"],
            "r2": test["r2"],
            "spearman": test["spearman"],
            "auroc_top3": test["auroc_top3"],
        },
        "model_path": model_path,
    }


def train_models(
    epochs=10,
    run_ablation=True,
    model_grid=None,
    output_file="output/models/training_results.json",
    run_metadata=None,
    write_output=True,
    gpu_id=None,
):
    (train_loader, val_loader, test_loader, graph_data,
     node_to_col_names_dict, node_to_col_stats,
     train_edge_index_dict, val_edge_index_dict, test_edge_index_dict,
     task) = prepare_data()

    device = get_device(gpu_id)
    graph_data = graph_data.to(device)

    # Move split-specific edge_index_dicts to the same device
    train_edge_index_dict = {et: ei.to(device) for et, ei in train_edge_index_dict.items()}
    val_edge_index_dict = {et: ei.to(device) for et, ei in val_edge_index_dict.items()}
    test_edge_index_dict = {et: ei.to(device) for et, ei in test_edge_index_dict.items()}

    if model_grid is None:
        model_grid = "high,zero,low" if run_ablation else "high,zero"
    selected_configs = parse_model_grid(model_grid)

    results = []
    for cfg_row in selected_configs:
        res = train_and_evaluate(
            cfg_row["name"],
            lambda_orthogonal=cfg_row["lambda_orthogonal"],
            aux_weight=cfg_row.get("aux_weight", 0.5),
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            graph_data=graph_data,
            node_to_col_names_dict=node_to_col_names_dict,
            node_to_col_stats=node_to_col_stats,
            device=device, epochs=epochs,
            train_edge_index_dict=train_edge_index_dict,
            val_edge_index_dict=val_edge_index_dict,
            test_edge_index_dict=test_edge_index_dict,
            task=task,
        )
        res["model_level"] = cfg_row["model_level"]
        if run_metadata:
            res["run_metadata"] = dict(run_metadata)
        results.append(res)

    best_model = sorted(
        results,
        key=lambda r: (r["test_metrics"]["mae"], r["test_metrics"]["orth"]),
    )[0]
    print(
        f"\nMelhor modelo (criterio: MAE asc, orth asc): "
        f"{best_model['model_name']}"
    )
    print(f"Metricas: {best_model['test_metrics']}")

    if write_output:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nResultados salvos em {output_file} e modelos em output/models/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F1 Pipeline (graph-only, cosine orthogonality)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip lambda ablation models")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of repeated runs for fixed split experiment")
    parser.add_argument("--seed_start", type=int, default=42,
                        help="Initial seed used for repeated runs")
    parser.add_argument(
        "--design",
        type=str,
        default="fixed_split_repeated_seeds",
        choices=["fixed_split_repeated_seeds"],
        help="Experimental design to execute from this script",
    )
    parser.add_argument(
        "--model_grid",
        type=str,
        default="zero,low,high",
        help="Comma-separated model levels. Valid: zero, low, high",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/models/training_results.json",
        help="Path to write run results JSON",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic torch backend behavior when seeding",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=cfg.DEFAULT_GPU_ID,
        help=f"CUDA device index (default: {cfg.DEFAULT_GPU_ID})",
    )
    args = parser.parse_args()

    all_results = []
    for run_idx in range(args.n_runs):
        seed = args.seed_start + run_idx
        print(f"\n=== Run {run_idx + 1}/{args.n_runs} | seed={seed} ===")
        set_global_seed(seed, deterministic=args.deterministic)
        run_results = train_models(
            epochs=args.epochs,
            run_ablation=not args.skip_ablation,
            model_grid=args.model_grid,
            output_file=args.output_file,
            run_metadata={"run_id": run_idx, "seed": seed, "design": args.design, "split_id": "default"},
            write_output=args.n_runs == 1,
            gpu_id=args.gpu_id,
        )
        all_results.extend(run_results)

    if args.n_runs > 1:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResultados agregados salvos em {args.output_file}")
