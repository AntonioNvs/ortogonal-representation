import torch
import torch.nn as nn


class DriverEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, out_dim=8, dropout_prob=0.3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim // 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x_driver):
        return self.mlp(x_driver)
