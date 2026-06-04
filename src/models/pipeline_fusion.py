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
    """Mean absolute cosine similarity between two batched tensors."""
    a_n = F.normalize(a, p=2, dim=-1)
    b_n = F.normalize(b, p=2, dim=-1)
    return torch.mean(torch.abs(torch.sum(a_n * b_n, dim=-1)))


class OrthogonalSeparationLoss(nn.Module):
    """
    Total loss = BCE + aux_weight * (BCE_piloto + BCE_equipe)
                 + lambda_orthogonal * cos|v_p, v_e|

    Single orthogonality strategy: cosine penalty between driver and team embeddings.
    """

    def __init__(self, lambda_orthogonal=0.0, aux_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_orthogonal = lambda_orthogonal
        self.aux_weight = aux_weight

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

        loss_orthogonal = pair_cosine(v_piloto, v_equipe)

        total_loss = loss_bce_total + self.lambda_orthogonal * loss_orthogonal

        return total_loss, loss_bce_total, loss_orthogonal
