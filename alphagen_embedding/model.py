from __future__ import annotations

import torch
from torch import Tensor, nn


class BehaviorEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: Tensor) -> Tensor:
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class StatEncoder(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, output_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HybridAlphaEmbeddingModel(nn.Module):
    def __init__(
        self,
        behavior_input_dim: int = 3,
        behavior_hidden_size: int = 32,
        stat_input_dim: int = 5,
        stat_hidden_dim: int = 16,
        embedding_dim: int = 16,
        behavior_layers: int = 1,
    ):
        super().__init__()
        self.behavior_encoder = BehaviorEncoder(
            input_dim=behavior_input_dim,
            hidden_size=behavior_hidden_size,
            num_layers=behavior_layers,
        )
        self.stat_encoder = StatEncoder(
            input_dim=stat_input_dim,
            hidden_dim=stat_hidden_dim,
            output_dim=stat_hidden_dim,
        )
        self.fuse = nn.Sequential(
            nn.Linear(behavior_hidden_size + stat_hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.weight_head = nn.Linear(embedding_dim, 1)

    def forward(self, behavior: Tensor, stats: Tensor, alpha_values: Tensor) -> dict[str, Tensor]:
        batch_size, num_alpha, lookback, feat_dim = behavior.shape
        behavior_flat = behavior.view(batch_size * num_alpha, lookback, feat_dim)
        stats_flat = stats.view(batch_size * num_alpha, stats.shape[-1])

        behavior_encoded = self.behavior_encoder(behavior_flat)
        stats_encoded = self.stat_encoder(stats_flat)
        fused = torch.cat([behavior_encoded, stats_encoded], dim=1)
        embeddings = self.fuse(fused).view(batch_size, num_alpha, -1)

        logits = self.weight_head(embeddings).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        prediction = (weights.unsqueeze(-1) * alpha_values).sum(dim=1)

        return {
            "embeddings": embeddings,
            "logits": logits,
            "weights": weights,
            "prediction": prediction,
        }
