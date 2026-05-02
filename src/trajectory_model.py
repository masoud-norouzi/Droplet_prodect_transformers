from __future__ import annotations

import torch
from torch import nn


class TrajectoryAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        t_future: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.t_future = t_future

        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, t_future * 2),
        )

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T_HISTORY, N_MAX, 4]
        # input_mask: [B, T_HISTORY, N_MAX]
        batch_size, t_history, n_max, _ = x.shape
        hidden = self.embed(x)
        hidden = hidden.reshape(batch_size * t_history, n_max, self.hidden_dim)

        padding_mask = ~input_mask.reshape(batch_size * t_history, n_max)
        hidden = hidden.transpose(0, 1)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        hidden = hidden.transpose(0, 1).reshape(batch_size, t_history, n_max, self.hidden_dim)

        mask = input_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = pooled / denom

        prediction = self.predictor(pooled)
        prediction = prediction.reshape(batch_size, n_max, self.t_future, 2)
        prediction = prediction.permute(0, 2, 1, 3)
        return prediction
