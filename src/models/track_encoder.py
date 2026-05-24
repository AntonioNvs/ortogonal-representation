import torch
import torch.nn as nn

class TrackEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, out_dim=16):
        super(TrackEncoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, x_track):
        return self.mlp(x_track)