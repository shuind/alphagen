from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_spearmanr
from alphagen_context.alpha_encoder import ContextAlphaEncoder, build_alpha_feature_inputs
from alphagen_context.ast_encoder import ASTGraphBuilder, ASTGraphEncoder
from alphagen_context.combiner import DeepSetsCombiner
from alphagen_context.utils import (
    combined_metric,
    expression_depth,
    expression_length,
    risky_operator_count,
    signal_turnover,
)


@dataclass
class ClusterInfo:
    cluster_id: int
    cluster_value: float
    cluster_coverage: float
    under_explore: float
    is_new: bool
    distance: float


class StructureClusterBank:
    def __init__(self, distance_threshold: float = 0.75, new_cluster_bonus: float = 0.05):
        self.distance_threshold = distance_threshold
        self.new_cluster_bonus = new_cluster_bonus
        self.centroids: List[Tensor] = []
        self.counts: List[int] = []
        self.mean_rewards: List[float] = []

    @property
    def total_count(self) -> int:
        return int(sum(self.counts))

    def describe(self, embedding: Tensor) -> ClusterInfo:
        if not self.centroids:
            return ClusterInfo(0, 0.0, 0.0, 1.0, True, 0.0)
        distances = [float(torch.norm(embedding - centroid).item()) for centroid in self.centroids]
        best_idx = int(min(range(len(distances)), key=lambda idx: distances[idx]))
        best_distance = distances[best_idx]
        is_new = best_distance > self.distance_threshold
        if is_new:
            return ClusterInfo(len(self.centroids), 0.0, 0.0, 1.0, True, best_distance)
        coverage = self.counts[best_idx] / max(1, self.total_count)
        return ClusterInfo(
            cluster_id=best_idx,
            cluster_value=float(self.mean_rewards[best_idx]),
            cluster_coverage=float(coverage),
            under_explore=float(1.0 - coverage),
            is_new=False,
            distance=best_distance,
        )

    def score(self, embedding: Tensor) -> ClusterInfo:
        return self.describe(embedding)

    def update(self, embedding: Tensor, reward: float) -> ClusterInfo:
        info = self.describe(embedding)
        if info.is_new:
            self.centroids.append(embedding.detach().cpu())
            self.counts.append(1)
            self.mean_rewards.append(float(reward))
            return ClusterInfo(
                cluster_id=len(self.centroids) - 1,
                cluster_value=float(reward),
                cluster_coverage=0.0,
                under_explore=1.0,
                is_new=True,
                distance=info.distance,
            )
        idx = info.cluster_id
        count = self.counts[idx]
        new_count = count + 1
        self.centroids[idx] = (self.centroids[idx] * count + embedding.detach().cpu()) / new_count
        self.counts[idx] = new_count
        self.mean_rewards[idx] = (self.mean_rewards[idx] * count + reward) / new_count
        coverage = self.counts[idx] / max(1, self.total_count)
        return ClusterInfo(
            cluster_id=idx,
            cluster_value=float(self.mean_rewards[idx]),
            cluster_coverage=float(coverage),
            under_explore=float(1.0 - coverage),
            is_new=False,
            distance=info.distance,
        )


class ContextEvaluator(nn.Module):
    def __init__(
        self,
        ast_encoder: ASTGraphEncoder,
        alpha_encoder: ContextAlphaEncoder,
        combiner: DeepSetsCombiner,
        graph_builder: Optional[ASTGraphBuilder] = None,
        lookback: int = 60,
        eta: float = 0.0,
        xi: float = 0.0,
        top_frac: float = 0.2,
        cluster_bank: Optional[StructureClusterBank] = None,
        ri_func_weight: float = 1.0,
        ri_struct_weight: float = 1.0,
        ri_reg_weight: float = 1.0,
        reward_lambda: float = 0.3,
        reward_schedule_decay: float = 0.0,
        reg_length_coef: float = 1e-3,
        reg_depth_coef: float = 2e-3,
        reg_risky_coef: float = 5e-3,
        reg_turnover_coef: float = 0.0,
    ):
        super().__init__()
        self.ast_encoder = ast_encoder
        self.alpha_encoder = alpha_encoder
        self.combiner = combiner
        self.graph_builder = graph_builder or ASTGraphBuilder()
        self.lookback = lookback
        self.eta = eta
        self.xi = xi
        self.top_frac = top_frac
        self.cluster_bank = cluster_bank or StructureClusterBank()
        self.ri_func_weight = ri_func_weight
        self.ri_struct_weight = ri_struct_weight
        self.ri_reg_weight = ri_reg_weight
        self.reward_lambda = reward_lambda
        self.reward_schedule_decay = reward_schedule_decay
        self.reg_length_coef = reg_length_coef
        self.reg_depth_coef = reg_depth_coef
        self.reg_risky_coef = reg_risky_coef
        self.reg_turnover_coef = reg_turnover_coef

    def current_lambda(self, step_idx: int) -> float:
        return float(self.reward_lambda / (1.0 + self.reward_schedule_decay * max(step_idx, 0)))

    @torch.no_grad()
    def encode_struct(self, exprs: Sequence[Expression]) -> Tensor:
        batch = self.graph_builder.build_batch(exprs)
        device = next(self.ast_encoder.parameters()).device
        return self.ast_encoder(batch.to(device))

    @torch.no_grad()
    def encode_alpha(self, alpha_panels: Tensor, target_values: Tensor) -> Tensor:
        behavior, stats = build_alpha_feature_inputs(
            alpha_panels,
            target_values,
            lookback=self.lookback,
            top_frac=self.top_frac,
        )
        device = next(self.alpha_encoder.parameters()).device
        return self.alpha_encoder(behavior.to(device), stats.to(device))

    @torch.no_grad()
    def combine_precomputed(
        self,
        embeddings: Tensor,
        alpha_panels: Tensor,
        target_values: Tensor,
    ) -> Dict[str, object]:
        if embeddings.shape[0] == 0:
            return {
                "metric": 0.0,
                "rankic": 0.0,
                "sharpe": 0.0,
                "turnover": 0.0,
                "weights": torch.zeros(0),
                "prediction": torch.zeros_like(target_values),
                "embeddings": torch.zeros((0, embeddings.shape[-1] if embeddings.ndim == 2 else 0)),
                "weight_sparsity": 0.0,
            }
        device = next(self.combiner.parameters()).device
        embeddings = embeddings.to(device)
        alpha_panels = alpha_panels.to(device)
        target_values = target_values.to(device)
        combine = self.combiner(embeddings.unsqueeze(0), alpha_panels.unsqueeze(0))
        prediction = combine["prediction"].squeeze(0)
        metrics = combined_metric(prediction, target_values, eta=self.eta, xi=self.xi, top_frac=self.top_frac)
        return {
            **metrics,
            "weights": combine["weights"].squeeze(0).detach().cpu(),
            "prediction": prediction.detach().cpu(),
            "embeddings": embeddings.detach().cpu(),
            "weight_sparsity": float(combine["weight_sparsity"].mean().item()),
        }

    @torch.no_grad()
    def evaluate_set(
        self,
        exprs: Sequence[Expression],
        alpha_panels: Tensor,
        target_values: Tensor,
    ) -> Dict[str, object]:
        if len(exprs) == 0:
            return {
                "metric": 0.0,
                "rankic": 0.0,
                "sharpe": 0.0,
                "turnover": 0.0,
                "weights": torch.zeros(0),
                "prediction": torch.zeros_like(target_values),
                "embeddings": torch.zeros((0, self.ast_encoder.output_dim + self.alpha_encoder.output_dim)),
                "weight_sparsity": 0.0,
            }
        device = next(self.combiner.parameters()).device
        alpha_panels = alpha_panels.to(device)
        target_values = target_values.to(device)
        z_struct = self.encode_struct(exprs)
        z_alpha = self.encode_alpha(alpha_panels, target_values)
        embeddings = torch.cat([z_alpha, z_struct], dim=-1)
        result = self.combine_precomputed(embeddings, alpha_panels, target_values)
        result["z_alpha"] = z_alpha.detach().cpu()
        result["z_struct"] = z_struct.detach().cpu()
        return result

    @torch.no_grad()
    def compute_ri_func(
        self,
        existing_panels: Tensor,
        candidate_panel: Tensor,
        target_values: Tensor,
    ) -> float:
        if existing_panels.numel() == 0 or existing_panels.shape[0] == 0:
            return float(batch_spearmanr(candidate_panel, target_values).mean().item())
        x = existing_panels.permute(1, 2, 0).reshape(-1, existing_panels.shape[0]).float()
        y = candidate_panel.reshape(-1).float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        beta = torch.linalg.lstsq(x, y).solution
        resid_panel = (y - x @ beta).view_as(candidate_panel)
        return float(batch_spearmanr(resid_panel, target_values).mean().item())

    @torch.no_grad()
    def compute_ri_struct_from_embedding(self, struct_embedding: Tensor, re_value: float = 0.0) -> tuple[float, ClusterInfo]:
        info = self.cluster_bank.score(struct_embedding.detach().cpu())
        base_value = info.cluster_value if not info.is_new else self.cluster_bank.new_cluster_bonus
        ri_struct = float(base_value * info.under_explore + (self.cluster_bank.new_cluster_bonus if info.is_new else 0.0))
        if re_value > 0:
            ri_struct += 0.1 * float(re_value) * info.under_explore
        return ri_struct, info

    @torch.no_grad()
    def compute_ri_struct(self, expr: Expression, re_value: float = 0.0) -> tuple[float, ClusterInfo]:
        struct_embedding = self.encode_struct([expr])[0].detach().cpu()
        return self.compute_ri_struct_from_embedding(struct_embedding, re_value=re_value)

    @torch.no_grad()
    def update_cluster_bank(self, expr: Expression, re_value: float) -> ClusterInfo:
        struct_embedding = self.encode_struct([expr])[0].detach().cpu()
        return self.cluster_bank.update(struct_embedding, re_value)

    @torch.no_grad()
    def update_cluster_bank_from_embedding(self, struct_embedding: Tensor, re_value: float) -> ClusterInfo:
        return self.cluster_bank.update(struct_embedding.detach().cpu(), re_value)

    def compute_ri_reg(self, expr: Expression, candidate_panel: Tensor) -> float:
        length_penalty = self.reg_length_coef * expression_length(expr)
        depth_penalty = self.reg_depth_coef * expression_depth(expr)
        risky_penalty = self.reg_risky_coef * risky_operator_count(expr)
        turnover_penalty = self.reg_turnover_coef * signal_turnover(candidate_panel, top_frac=self.top_frac)
        return float(-(length_penalty + depth_penalty + risky_penalty + turnover_penalty))

    def compose_reward(
        self,
        step_idx: int,
        re_value: float,
        ri_func: float,
        ri_struct: float,
        ri_reg: float,
    ) -> tuple[float, float]:
        reward_lambda = self.current_lambda(step_idx)
        ri_total = (
            self.ri_func_weight * ri_func
            + self.ri_struct_weight * ri_struct
            + self.ri_reg_weight * ri_reg
        )
        reward = re_value + reward_lambda * ri_total
        return float(reward), float(reward_lambda)
