from __future__ import annotations

import torch
from torch import nn


class TrajectoryAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        numeric_input_dim: int = 6,
        num_channel_embeddings: int = 1,
        channel_embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        t_future: int = 10,
        target_dim: int = 2,
    ) -> None:
        super().__init__()
        if input_dim != numeric_input_dim + 1:
            raise ValueError(
                "input_dim must equal numeric_input_dim plus one channel_id feature."
            )

        self.input_dim = input_dim
        self.numeric_input_dim = numeric_input_dim
        self.num_channel_embeddings = num_channel_embeddings
        self.hidden_dim = hidden_dim
        self.t_future = t_future
        self.target_dim = target_dim

        self.numeric_embed = nn.Sequential(
            nn.Linear(numeric_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.channel_embed = nn.Embedding(num_channel_embeddings, channel_embedding_dim)
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim + channel_embedding_dim, hidden_dim),
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
            nn.Linear(hidden_dim, t_future * target_dim),
        )

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T_HISTORY, N_MAX, input_dim]
        # final feature is channel_id_int
        # input_mask: [B, T_HISTORY, N_MAX]
        batch_size, t_history, n_max, _ = x.shape
        numeric_x = x[..., : self.numeric_input_dim]
        channel_ids = x[..., self.numeric_input_dim].long()
        channel_ids = channel_ids.clamp(min=0, max=self.num_channel_embeddings - 1)

        numeric_hidden = self.numeric_embed(numeric_x)
        channel_hidden = self.channel_embed(channel_ids)
        hidden = self.embed(torch.cat([numeric_hidden, channel_hidden], dim=-1))
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
        prediction = prediction.reshape(batch_size, n_max, self.t_future, self.target_dim)
        prediction = prediction.permute(0, 2, 1, 3)
        return prediction
