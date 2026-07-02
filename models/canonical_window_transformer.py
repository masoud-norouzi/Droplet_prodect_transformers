from __future__ import annotations

import torch
from torch import nn


class CanonicalWindowTransformer(nn.Module):
    def __init__(
        self,
        input_dim=5,
        target_dim=2,
        T_history=20,
        T_future=10,
        max_droplets=64,
        d_model=128,
        n_heads=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.T_history = T_history
        self.T_future = T_future
        self.max_droplets = max_droplets
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.droplet_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        self.time_embedding = nn.Embedding(T_history, d_model)
        self.slot_embedding = nn.Embedding(max_droplets, d_model)
        self.mask_embedding = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, T_future * target_dim),
        )

    def forward(
        self,
        history_x: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, M, F = history_x.shape
        assert T == self.T_history
        assert M == self.max_droplets
        assert F == self.input_dim
        assert history_mask.shape == (B, T, M)

        h = self.droplet_mlp(history_x)

        time_ids = torch.arange(T, device=history_x.device)
        slot_ids = torch.arange(M, device=history_x.device)
        time_embedding = self.time_embedding(time_ids).view(1, T, 1, self.d_model)
        slot_embedding = self.slot_embedding(slot_ids).view(1, 1, M, self.d_model)
        mask_embedding = self.mask_embedding(history_mask.long())

        h = h + time_embedding + slot_embedding + mask_embedding
        h = h.reshape(B, T * M, self.d_model)

        src_key_padding_mask = (~history_mask).reshape(B, T * M)
        h = self.transformer(
            h,
            src_key_padding_mask=src_key_padding_mask,
        )

        h = h.reshape(B, T, M, self.d_model)
        h_last = h[:, -1, :, :]

        pred_v = self.velocity_head(h_last)
        pred_v = pred_v.reshape(B, M, self.T_future, self.target_dim)
        pred_v = pred_v.permute(0, 2, 1, 3).contiguous()

        return pred_v
