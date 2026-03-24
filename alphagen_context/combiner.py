from __future__ import annotations

import torch
from torch import Tensor, nn


def sparsemax(input: Tensor, dim: int = -1) -> Tensor:
    shifted = input - input.max(dim=dim, keepdim=True).values
    sorted_input, _ = torch.sort(shifted, descending=True, dim=dim)
    cumsum = sorted_input.cumsum(dim) - 1
    rhos = torch.arange(1, sorted_input.size(dim) + 1, device=input.device, dtype=input.dtype)
    view_shape = [1] * input.ndim
    view_shape[dim] = -1
    rhos = rhos.view(view_shape)
    support = sorted_input > cumsum / rhos
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cumsum.gather(dim, k - 1) / k.to(input.dtype)
    return torch.clamp(shifted - tau, min=0.0)


class DeepSetsCombiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        activation: str = "sparsemax",
    ):
        super().__init__()
        self.activation = activation
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.psi = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: Tensor, alpha_values: Tensor) -> dict[str, Tensor]:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must be [batch, num_alpha, dim]")
        if alpha_values.ndim < 3:
            raise ValueError("alpha_values must be [batch, num_alpha, ...]")
        phi_values = self.phi(embeddings)
        context = phi_values.mean(dim=1, keepdim=True).expand_as(phi_values)
        logits = self.psi(torch.cat([embeddings, context], dim=-1)).squeeze(-1)
        if self.activation == "softmax":
            weights = torch.softmax(logits, dim=1)
        else:
            weights = sparsemax(logits, dim=1)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        expand_shape = [weights.shape[0], weights.shape[1]] + [1] * (alpha_values.ndim - 2)
        prediction = (alpha_values * weights.view(*expand_shape)).sum(dim=1)
        sparsity = (weights.abs() < 1e-4).float().mean(dim=1)
        return {
            "weights": weights,
            "logits": logits,
            "prediction": prediction,
            "weight_sparsity": sparsity,
        }
