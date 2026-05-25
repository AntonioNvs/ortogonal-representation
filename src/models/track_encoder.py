import torch
import torch.nn as nn


class TrackEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, out_dim=8, dropout_prob=0.2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x_track):
        return self.mlp(x_track)
