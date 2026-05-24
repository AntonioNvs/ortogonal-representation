import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driver_encoder import DriverEncoder
from models.team_encoder import TeamGraphEncoder
from models.track_encoder import TrackEncoder

class F1OrthogonalPipeline(nn.Module):
    def __init__(self, num_nodes_dict, latent_dim=8, use_track_encoder=True):
        super(F1OrthogonalPipeline, self).__init__()
        
        self.use_track_encoder = use_track_encoder
        self.driver_encoder = DriverEncoder(out_dim=latent_dim)
        self.team_encoder = TeamGraphEncoder(num_nodes_dict=num_nodes_dict, out_dim=latent_dim)
        
        if self.use_track_encoder:
            self.track_encoder = TrackEncoder(out_dim=latent_dim)
            classifier_input_dim = latent_dim * 3
        else:
            classifier_input_dim = latent_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.aux_piloto = nn.Linear(latent_dim, 1)
        self.aux_equipe = nn.Linear(latent_dim, 1)
        
    def forward(self, x_driver, x_track, graph_x_dict, graph_edge_index_dict, target_constructor_ids):
        v_piloto = self.driver_encoder(x_driver)
        v_equipe = self.team_encoder(graph_x_dict, graph_edge_index_dict, target_constructor_ids)
        
        if self.use_track_encoder:
            v_pista = self.track_encoder(x_track)
            v_fused = torch.cat([v_piloto, v_equipe, v_pista], dim=-1)
        else:
            v_fused = torch.cat([v_piloto, v_equipe], dim=-1)
        
        logits = self.classifier(v_fused)
        logits_piloto = self.aux_piloto(v_piloto)
        logits_equipe = self.aux_equipe(v_equipe)
        
        return logits, logits_piloto, logits_equipe, v_piloto, v_equipe

class OrthogonalSeparationLoss(nn.Module):
    def __init__(self, lambda_orthogonal=0.01):
        super(OrthogonalSeparationLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_orth = lambda_orthogonal
        
    def forward(self, logits, logits_piloto, logits_equipe, targets, v_piloto, v_equipe):
        loss_bce_main = self.bce(logits.squeeze(-1), targets.float())
        loss_bce_piloto = self.bce(logits_piloto.squeeze(-1), targets.float())
        loss_bce_equipe = self.bce(logits_equipe.squeeze(-1), targets.float())
        
        loss_bce = loss_bce_main + 0.5 * loss_bce_piloto + 0.5 * loss_bce_equipe
        
        v_piloto_norm = F.normalize(v_piloto, p=2, dim=-1)
        v_equipe_norm = F.normalize(v_equipe, p=2, dim=-1)
        
        cosine_sim = torch.sum(v_piloto_norm * v_equipe_norm, dim=-1)
        
        loss_orthogonal = torch.mean(torch.abs(cosine_sim))
        
        total_loss = loss_bce + (self.lambda_orth * loss_orthogonal)
        
        return total_loss, loss_bce_main, loss_orthogonal