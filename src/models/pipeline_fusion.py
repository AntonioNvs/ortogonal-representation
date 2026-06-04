import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_encoder import HeteroGraphEncoder


class F1OrthogonalPipeline(nn.Module):
    """
    Pipeline with a single HeteroGraphEncoder producing both driver and team
    embeddings, fused via concatenation and fed into a classifier + aux heads.
    """

    def __init__(self, num_nodes_dict, latent_dim=8):
        super().__init__()

        self.graph_encoder = HeteroGraphEncoder(
            num_nodes_dict=num_nodes_dict,
            out_dim=latent_dim,
        )

        classifier_input_dim = latent_dim * 2  # v_piloto + v_equipe

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.aux_piloto = nn.Linear(latent_dim, 1)
        self.aux_equipe = nn.Linear(latent_dim, 1)

    def forward(self, graph_x_dict, graph_edge_index_dict,
                target_constructor_ids, target_driver_ids):
        out_dict = self.graph_encoder(graph_x_dict, graph_edge_index_dict)

        v_equipe = out_dict["constructors"][target_constructor_ids]
        v_piloto = out_dict["driver"][target_driver_ids]

        v_fused = torch.cat([v_piloto, v_equipe], dim=-1)

        logits = self.classifier(v_fused)
        logits_piloto = self.aux_piloto(v_piloto)
        logits_equipe = self.aux_equipe(v_equipe)

        return logits, logits_piloto, logits_equipe, v_piloto, v_equipe


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


class OrthogonalSeparationLoss(nn.Module):
    """
    Total loss = BCE + aux_weight * (BCE_piloto + BCE_equipe)
                 + lambda_orthogonal * loss_orthogonal

    Modos de ortogonalidade (parametro mode):
      - "pair" (default)     : cos(v_p[i], v_e[i]) pareado
      - "all_pairs"          : cos(v_p[i], v_e[j]) para todo par (i,j)
      - "cross_corr"         : matriz de correlacao cruzada (d, d)
    """

    def __init__(self, lambda_orthogonal=0.0, aux_weight=0.5, mode=ORTH_MODE_PAIR):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_orthogonal = lambda_orthogonal
        self.aux_weight = aux_weight
        self.mode = mode

    def forward(self, logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe):
        loss_bce_main = self.bce(logits.squeeze(-1), targets.float())
        loss_bce_piloto = self.bce(logits_piloto.squeeze(-1), targets.float())
        loss_bce_equipe = self.bce(logits_equipe.squeeze(-1), targets.float())

        loss_bce_total = (
            loss_bce_main
            + self.aux_weight * loss_bce_piloto
            + self.aux_weight * loss_bce_equipe
        )

        if self.mode == ORTH_MODE_ALLPAIRS:
            loss_orthogonal = all_pairs_cosine(v_piloto, v_equipe)
        elif self.mode == ORTH_MODE_CROSSCORR:
            loss_orthogonal = cross_correlation_loss(v_piloto, v_equipe)
        else:
            loss_orthogonal = pair_cosine(v_piloto, v_equipe)

        total_loss = loss_bce_total + self.lambda_orthogonal * loss_orthogonal

        return total_loss, loss_bce_total, loss_orthogonal
