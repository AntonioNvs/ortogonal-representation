import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_encoder import HeteroGraphEncoder
from relbench.modeling.nn import HeteroEncoder


class F1OrthogonalPipeline(nn.Module):
    """
    Pipeline with a single HeteroGraphEncoder producing both driver and team
    embeddings, fused via concatenation and fed into a classifier + aux heads.

    Pre-race features (qualifying_position, grid) are concatenated with the
    fused [driver || constructor] embedding before the classifier, so the
    model learns: given qualifying position X, grid position G, driver D,
    constructor C → predicted race finishing position.
    """

    def __init__(self, num_nodes_dict, latent_dim=8, node_to_col_names_dict=None, node_to_col_stats=None, **kwargs):
        super().__init__()

        # Try loading metadata from saved cache if not passed directly
        meta_path = "output/models/graph_meta.pt"
        if (node_to_col_names_dict is None or node_to_col_stats is None) and os.path.exists(meta_path):
            try:
                meta = torch.load(meta_path, map_location="cpu")
                node_to_col_names_dict = meta.get("node_to_col_names_dict")
                node_to_col_stats = meta.get("node_to_col_stats")
            except Exception as e:
                print(f"Warning: Failed to load graph metadata from {meta_path}: {e}")

        if node_to_col_names_dict is not None and node_to_col_stats is not None:
            self.encoder = HeteroEncoder(
                channels=latent_dim,
                node_to_col_names_dict=node_to_col_names_dict,
                node_to_col_stats=node_to_col_stats,
            )
        else:
            self.encoder = None

        self.graph_encoder = HeteroGraphEncoder(
            num_nodes_dict=num_nodes_dict,
            out_dim=latent_dim,
        )

        # 2 extra dims for qualifying_position + grid (pre-race features)
        classifier_input_dim = latent_dim * 2 + 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.aux_piloto = nn.Linear(latent_dim, 1)
        self.aux_equipe = nn.Linear(latent_dim, 1)

    def forward(self, graph_x_dict, graph_edge_index_dict,
                target_constructor_ids, target_driver_ids,
                qualifying_position=None, grid=None,
                graph_tf_dict=None):
        if (graph_x_dict is None or len(graph_x_dict) == 0) and graph_tf_dict is not None and self.encoder is not None:
            graph_x_dict = self.encoder(graph_tf_dict)

        out_dict = self.graph_encoder(graph_x_dict, graph_edge_index_dict)

        v_equipe = out_dict["constructors"][target_constructor_ids]
        v_piloto = out_dict["drivers"][target_driver_ids]

        v_fused = torch.cat([v_piloto, v_equipe], dim=-1)

        # Concatenate pre-race features (qualifying position + grid)
        if qualifying_position is not None:
            qualifying_position = qualifying_position.unsqueeze(-1)  # (B, 1)
            v_fused = torch.cat([v_fused, qualifying_position], dim=-1)
        if grid is not None:
            grid = grid.unsqueeze(-1)  # (B, 1)
            v_fused = torch.cat([v_fused, grid], dim=-1)

        logits = self.classifier(v_fused)
        logits_piloto = self.aux_piloto(v_piloto)
        logits_equipe = self.aux_equipe(v_equipe)

        paired_orthogonal_loss = self.graph_encoder.compute_paired_orthogonal_loss(
            out_dict, graph_edge_index_dict
        )

        return logits, logits_piloto, logits_equipe, v_piloto, v_equipe, paired_orthogonal_loss


def pair_cosine(a, b):
    """Mean absolute cosine similarity between paired samples (a[i], b[i])."""
    a_n = F.normalize(a, p=2, dim=-1)
    b_n = F.normalize(b, p=2, dim=-1)
    return torch.mean(torch.abs(torch.sum(a_n * b_n, dim=-1)))


def all_pairs_cosine(a, b):
    """
    Cosseno par-a-par entre TODOS os embeddings de piloto e TODOS de equipe.

    Em vez de medir cos(v_p[i], v_e[i]) para cada par (i),
    calcula a matriz NxN de cossenos e tira a media do valor absoluto.
    Se os espacos estao bem separados, mesmo pares cruzados (i,j) tem cosseno baixo.
    """
    a_n = F.normalize(a, p=2, dim=-1)  # (N, d)
    b_n = F.normalize(b, p=2, dim=-1)  # (N, d)
    cos_mat = torch.mm(a_n, b_n.T)     # (N, N) — cosseno entre todo par
    return torch.mean(torch.abs(cos_mat))


def cross_correlation_loss(a, b, eps=1e-8):
    """
    Perda baseada na matriz de correlacao cruzada dimensao-a-dimensao.

    Para cada par de dimensoes (d_p, d_e), calcula a correlacao linear
    atraves das amostras do batch. Penaliza o valor absoluto medio
    de TODA a matriz d x d.

    Inspirado em Barlow Twins / VICReg: em vez de separar amostras,
    forcamos as dimensoes latentes de piloto e equipe a serem
    linearmente independentes entre si.

    Diferenca crucial vs cosseno pareado:
      - cosseno pareado:  media de cos(v_p[i], v_e[i]) — baixo quando
        cada par individual e ortogonal
      - correlacao cruzada: media de |corr(d_p, d_e)| — baixo quando
        NAO HA relacao linear entre nenhuma dimensao de piloto e equipe
    """
    a_c = a - a.mean(dim=0, keepdim=True)      # centralizar
    b_c = b - b.mean(dim=0, keepdim=True)
    a_n = a_c / (a_c.std(dim=0, keepdim=True) + eps)
    b_n = b_c / (b_c.std(dim=0, keepdim=True) + eps)

    n = a.shape[0]
    cross_corr = torch.mm(a_n.T, b_n) / (n - 1)   # (d, d)
    return torch.mean(torch.abs(cross_corr))


ORTH_MODE_PAIR = "pair"
ORTH_MODE_ALLPAIRS = "all_pairs"
ORTH_MODE_CROSSCORR = "cross_corr"
ORTH_MODE_PAIRED_DRIVER_CONSTRUCTOR = "paired_driver_constructor"

TASK_KIND_CLASSIFICATION = "classification"
TASK_KIND_REGRESSION = "regression"


class OrthogonalSeparationLoss(nn.Module):
    """
    Total loss = L_main + aux_weight * (L_piloto + L_equipe)
                 + lambda_orthogonal * loss_orthogonal

    ``task`` selects the per-head criterion L:
      - "classification" (default): BCEWithLogitsLoss, targets in {0, 1}
        (e.g. driver-top3).
      - "regression": SmoothL1Loss (Huber), targets are continuous/ordinal
        (e.g. results-position, results-positionorder, results-points).
        SmoothL1 is used instead of plain MSE because finishing
        position/points targets have occasional large outliers (DNFs coded
        as a high positionOrder, retirements at points=0 after a strong
        grid slot) that would otherwise dominate the gradient.

    Modos de ortogonalidade (parametro mode):
      - "pair" (default)     : cos(v_p[i], v_e[i]) pareado
      - "all_pairs"          : cos(v_p[i], v_e[j]) para todo par (i,j)
      - "cross_corr"         : matriz de correlacao cruzada (d, d)
      - "paired_driver_constructor" : loss para pares de driver e constructor
    """

    def __init__(self, lambda_orthogonal=0.0, aux_weight=0.5, mode=ORTH_MODE_PAIR, task=TASK_KIND_CLASSIFICATION):
        super().__init__()
        self.task = task
        if task == TASK_KIND_REGRESSION:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.lambda_orthogonal = lambda_orthogonal
        self.aux_weight = aux_weight
        self.mode = mode

    def forward(self, logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe, paired_orthogonal_loss=None):
        loss_main = self.criterion(logits.squeeze(-1), targets.float())
        loss_piloto = self.criterion(logits_piloto.squeeze(-1), targets.float())
        loss_equipe = self.criterion(logits_equipe.squeeze(-1), targets.float())

        loss_total = (
            loss_main
            + self.aux_weight * loss_piloto
            + self.aux_weight * loss_equipe
        )

        if self.mode == ORTH_MODE_ALLPAIRS:
            loss_orthogonal = all_pairs_cosine(v_piloto, v_equipe)
        elif self.mode == ORTH_MODE_CROSSCORR:
            loss_orthogonal = cross_correlation_loss(v_piloto, v_equipe)
        elif self.mode == ORTH_MODE_PAIRED_DRIVER_CONSTRUCTOR:
            if paired_orthogonal_loss is None:
                raise ValueError(
                    "paired_orthogonal_loss must be provided when using "
                    f"mode='{ORTH_MODE_PAIRED_DRIVER_CONSTRUCTOR}'."
                )
            loss_orthogonal = paired_orthogonal_loss
        else:
            loss_orthogonal = pair_cosine(v_piloto, v_equipe)

        total_loss = loss_total + self.lambda_orthogonal * loss_orthogonal

        return total_loss, loss_total, loss_orthogonal
