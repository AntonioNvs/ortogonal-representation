import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driver_encoder import DriverEncoder
from models.team_encoder import TeamGraphEncoder
from models.track_encoder import TrackEncoder


def _hsic_rbf(x, y, sigma_x=None, sigma_y=None):
    """Biased HSIC with RBF kernels, used as a differentiable regularizer."""
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


class F1OrthogonalPipeline(nn.Module):
    def __init__(self, num_nodes_dict, latent_dim=8, driver_input_dim=6,
                 track_input_dim=5, use_track_encoder=True):
        super().__init__()

        self.use_track_encoder = use_track_encoder
        self.driver_encoder = DriverEncoder(input_dim=driver_input_dim, out_dim=latent_dim)
        self.team_encoder = TeamGraphEncoder(num_nodes_dict=num_nodes_dict, out_dim=latent_dim)

        if self.use_track_encoder:
            self.track_encoder = TrackEncoder(input_dim=track_input_dim, out_dim=latent_dim)
            classifier_input_dim = latent_dim * 3
        else:
            classifier_input_dim = latent_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.aux_piloto = nn.Linear(latent_dim, 1)
        self.aux_equipe = nn.Linear(latent_dim, 1)

    def forward(self, x_driver, x_track, graph_x_dict, graph_edge_index_dict,
                target_constructor_ids):
        v_piloto = self.driver_encoder(x_driver)
        v_equipe = self.team_encoder(graph_x_dict, graph_edge_index_dict, target_constructor_ids)

        if self.use_track_encoder:
            v_pista = self.track_encoder(x_track)
            v_fused = torch.cat([v_piloto, v_equipe, v_pista], dim=-1)
        else:
            v_pista = None
            v_fused = torch.cat([v_piloto, v_equipe], dim=-1)

        logits = self.classifier(v_fused)
        logits_piloto = self.aux_piloto(v_piloto)
        logits_equipe = self.aux_equipe(v_equipe)

        return logits, logits_piloto, logits_equipe, v_piloto, v_equipe, v_pista


def _pair_cosine(a, b):
    a_n = F.normalize(a, p=2, dim=-1)
    b_n = F.normalize(b, p=2, dim=-1)
    return torch.mean(torch.abs(torch.sum(a_n * b_n, dim=-1)))


def _pair_cross_corr(a, b):
    eps = 1e-8
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    a_s = a_c / (a_c.std(dim=0, keepdim=True) + eps)
    b_s = b_c / (b_c.std(dim=0, keepdim=True) + eps)
    n = a.size(0)
    cross = torch.matmul(a_s.T, b_s) / max(n - 1, 1)
    return torch.mean(torch.abs(cross))


class OrthogonalSeparationLoss(nn.Module):
    """
    Total loss = BCE_main + aux_weight * (BCE_piloto + BCE_equipe)
               + lambda_pairwise * sum cos|pairs|
               + lambda_crossdim * sum |cross-corr|
               + lambda_hsic    * sum HSIC(pairs)

    'pairs' covers (piloto,equipe) and optionally (piloto,pista),(equipe,pista)
    when v_pista is provided.
    """

    def __init__(self, lambda_orthogonal=0.0, lambda_crossdim=0.0, lambda_hsic=0.0,
                 aux_weight=0.5, include_track_pairs=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_pairwise = lambda_orthogonal
        self.lambda_crossdim = lambda_crossdim
        self.lambda_hsic = lambda_hsic
        self.aux_weight = aux_weight
        self.include_track_pairs = include_track_pairs

    def forward(self, logits, logits_piloto, logits_equipe, targets,
                v_piloto, v_equipe, v_pista=None):
        loss_bce_main = self.bce(logits.squeeze(-1), targets.float())
        loss_bce_piloto = self.bce(logits_piloto.squeeze(-1), targets.float())
        loss_bce_equipe = self.bce(logits_equipe.squeeze(-1), targets.float())

        loss_bce = loss_bce_main + self.aux_weight * loss_bce_piloto + self.aux_weight * loss_bce_equipe

        pairs = [(v_piloto, v_equipe)]
        if self.include_track_pairs and v_pista is not None:
            pairs.append((v_piloto, v_pista))
            pairs.append((v_equipe, v_pista))

        loss_orthogonal = sum(_pair_cosine(a, b) for a, b in pairs) / len(pairs)
        loss_crossdim = sum(_pair_cross_corr(a, b) for a, b in pairs) / len(pairs)
        loss_hsic = sum(_hsic_rbf(a, b) for a, b in pairs) / len(pairs)

        total_loss = (
            loss_bce
            + self.lambda_pairwise * loss_orthogonal
            + self.lambda_crossdim * loss_crossdim
            + self.lambda_hsic * loss_hsic
        )

        return total_loss, loss_bce_main, loss_orthogonal, loss_crossdim, loss_hsic
