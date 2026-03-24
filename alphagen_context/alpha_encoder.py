from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_embedding.model import BehaviorEncoder, StatEncoder


def _daily_top_bottom_return(alpha_panel: Tensor, target_panel: Tensor, top_frac: float = 0.2) -> Tensor:
    days, stocks = alpha_panel.shape
    k = max(1, int(stocks * top_frac))
    returns = torch.zeros(days, dtype=alpha_panel.dtype, device=alpha_panel.device)
    for day in range(days):
        alpha_day = alpha_panel[day]
        target_day = target_panel[day]
        top_idx = torch.topk(alpha_day, k=k, largest=True).indices
        bottom_idx = torch.topk(alpha_day, k=k, largest=False).indices
        returns[day] = target_day[top_idx].mean() - target_day[bottom_idx].mean()
    return returns


def _daily_turnover(alpha_panel: Tensor, top_frac: float = 0.2) -> Tensor:
    days, stocks = alpha_panel.shape
    if days <= 1:
        return torch.zeros(days, dtype=alpha_panel.dtype, device=alpha_panel.device)
    k = max(1, int(stocks * top_frac))
    positions = torch.zeros_like(alpha_panel)
    for day in range(days):
        alpha_day = alpha_panel[day]
        top_idx = torch.topk(alpha_day, k=k, largest=True).indices
        bottom_idx = torch.topk(alpha_day, k=k, largest=False).indices
        positions[day, top_idx] = 1.0
        positions[day, bottom_idx] = -1.0
    diffs = (positions[1:] - positions[:-1]).abs().mean(dim=1)
    return torch.cat([torch.zeros(1, dtype=alpha_panel.dtype, device=alpha_panel.device), diffs], dim=0)


def build_alpha_feature_inputs(
    alpha_values: Tensor,
    target_values: Tensor,
    lookback: int = 60,
    top_frac: float = 0.2,
) -> Tuple[Tensor, Tensor]:
    if alpha_values.ndim != 3:
        raise ValueError("alpha_values must be [num_alpha, days, stocks]")
    num_alpha, days, _ = alpha_values.shape
    if target_values.shape != alpha_values.shape[1:]:
        raise ValueError("target_values must be [days, stocks]")
    if days < lookback:
        raise ValueError(f"Need at least {lookback} days, got {days}")

    ic_daily = []
    rankic_daily = []
    return_daily = []
    turnover_daily = []
    for idx in range(num_alpha):
        panel = alpha_values[idx]
        ic_daily.append(torch.nan_to_num(batch_pearsonr(panel, target_values), nan=0.0, posinf=0.0, neginf=0.0))
        rankic_daily.append(torch.nan_to_num(batch_spearmanr(panel, target_values), nan=0.0, posinf=0.0, neginf=0.0))
        return_daily.append(torch.nan_to_num(_daily_top_bottom_return(panel, target_values, top_frac=top_frac), nan=0.0, posinf=0.0, neginf=0.0))
        turnover_daily.append(torch.nan_to_num(_daily_turnover(panel, top_frac=top_frac), nan=0.0, posinf=0.0, neginf=0.0))

    ic_tensor = torch.stack(ic_daily, dim=0)
    rankic_tensor = torch.stack(rankic_daily, dim=0)
    return_tensor = torch.stack(return_daily, dim=0)
    turnover_tensor = torch.stack(turnover_daily, dim=0)

    behavior = torch.stack(
        [
            ic_tensor[:, -lookback:],
            rankic_tensor[:, -lookback:],
            return_tensor[:, -lookback:],
        ],
        dim=2,
    )
    behavior = torch.nan_to_num(behavior, nan=0.0, posinf=0.0, neginf=0.0)
    stats = torch.stack(
        [
            ic_tensor.mean(dim=1),
            ic_tensor.std(dim=1, unbiased=False),
            return_tensor.mean(dim=1),
            return_tensor.std(dim=1, unbiased=False),
            turnover_tensor.mean(dim=1),
        ],
        dim=1,
    )
    stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    return behavior.float(), stats.float()


class ContextAlphaEncoder(nn.Module):
    def __init__(
        self,
        behavior_input_dim: int = 3,
        behavior_hidden_size: int = 32,
        stat_input_dim: int = 5,
        stat_hidden_dim: int = 16,
        output_dim: int = 32,
        behavior_layers: int = 1,
    ):
        super().__init__()
        self.output_dim = output_dim
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
            nn.Linear(behavior_hidden_size + stat_hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, behavior: Tensor, stats: Tensor) -> Tensor:
        added_batch = False
        if behavior.ndim == 3:
            behavior = behavior.unsqueeze(0)
            stats = stats.unsqueeze(0)
            added_batch = True
        batch_size, num_alpha, lookback, feat_dim = behavior.shape
        behavior_flat = behavior.view(batch_size * num_alpha, lookback, feat_dim)
        stats_flat = stats.view(batch_size * num_alpha, stats.shape[-1])
        behavior_encoded = self.behavior_encoder(behavior_flat)
        stats_encoded = self.stat_encoder(stats_flat)
        fused = self.fuse(torch.cat([behavior_encoded, stats_encoded], dim=1))
        fused = fused.view(batch_size, num_alpha, -1)
        return fused.squeeze(0) if added_batch else fused
