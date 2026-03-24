from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from alphagen.data.expression import Expression, OutOfDataRangeError
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen_context.evaluator import ContextEvaluator
from alphagen_qlib.calculator import QLibStockDataCalculator


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


class ContextAlphaPool(AlphaPoolBase):
    def __init__(
        self,
        capacity: int,
        calculator: QLibStockDataCalculator,
        evaluator: ContextEvaluator,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(capacity=capacity, calculator=calculator, device=device)
        self.evaluator = evaluator
        self.exprs: List[Expression] = []
        self.alpha_panels: List[Tensor] = []
        self.weights: np.ndarray = np.zeros(capacity + 1, dtype=np.float32)
        self.best_metric: float = -math.inf
        self.best_rankic: float = -math.inf
        self.eval_cnt: int = 0
        self.last_reward_info: Dict[str, float] = {}
        self.last_cluster_info: Dict[str, float | int | bool] = {}
        self._target_panel_cache: Optional[Tensor] = None
        self._current_alpha_tensor: Optional[Tensor] = None
        self._current_struct_embeddings: Optional[Tensor] = None
        self._current_alpha_embeddings: Optional[Tensor] = None
        self._current_eval_result: Optional[Dict[str, object]] = None
        self._panel_cache: Dict[str, Tensor] = {}
        self._struct_cache: Dict[str, Tensor] = {}
        self._alpha_cache: Dict[str, Tensor] = {}

    @property
    def size(self) -> int:
        return len(self.exprs)

    @property
    def state(self) -> dict:
        return {
            "exprs": list(self.exprs),
            "weights": list(self.weights[: self.size]),
            "best_metric": self.best_metric,
            "best_rankic": self.best_rankic,
        }

    def to_dict(self) -> dict:
        return {
            "exprs": [str(expr) for expr in self.exprs],
            "weights": [float(x) for x in self.weights[: self.size]],
            "best_metric": float(self.best_metric),
            "best_rankic": float(self.best_rankic),
            "last_reward_info": _to_builtin(self.last_reward_info),
            "last_cluster_info": _to_builtin(self.last_cluster_info),
        }

    def _target_panel(self) -> Tensor:
        if self._target_panel_cache is not None:
            return self._target_panel_cache
        if self.calculator.target_value is None:
            raise RuntimeError("ContextAlphaPool requires calculator.target_value")
        self._target_panel_cache = self.calculator.target_value.detach().cpu()
        return self._target_panel_cache

    def _expr_key(self, expr: Expression) -> str:
        return str(expr)

    def _evaluate_expr(self, expr: Expression) -> Tensor:
        expr_key = self._expr_key(expr)
        cached = self._panel_cache.get(expr_key)
        if cached is not None:
            return cached
        panel = self.calculator._calc_alpha(expr).detach().cpu()
        self._panel_cache[expr_key] = panel
        return panel

    def _encode_struct_cached(self, expr: Expression) -> Tensor:
        expr_key = self._expr_key(expr)
        cached = self._struct_cache.get(expr_key)
        if cached is not None:
            return cached
        struct = self.evaluator.encode_struct([expr])[0].detach().cpu()
        self._struct_cache[expr_key] = struct
        return struct

    def _encode_alpha_cached(self, expr: Expression, panel: Tensor) -> Tensor:
        expr_key = self._expr_key(expr)
        cached = self._alpha_cache.get(expr_key)
        if cached is not None:
            return cached
        alpha = self.evaluator.encode_alpha(panel.unsqueeze(0), self._target_panel()).detach().cpu()[0]
        self._alpha_cache[expr_key] = alpha
        return alpha

    def _stack_panels(self, panels: Sequence[Tensor]) -> Tensor:
        return torch.stack(list(panels), dim=0)

    def _evaluate_current(self) -> Dict[str, object]:
        if self._current_eval_result is not None:
            return self._current_eval_result
        if not self.exprs:
            return {
                "metric": 0.0,
                "rankic": 0.0,
                "weights": torch.zeros(0),
                "prediction": torch.zeros_like(self._target_panel()),
                "weight_sparsity": 0.0,
            }
        return self._refresh_cached_state()

    def _refresh_cached_state(self) -> Dict[str, object]:
        if not self.exprs:
            self._current_alpha_tensor = None
            self._current_alpha_embeddings = None
            self._current_struct_embeddings = None
            self._current_eval_result = {
                "metric": 0.0,
                "rankic": 0.0,
                "weights": torch.zeros(0),
                "prediction": torch.zeros_like(self._target_panel()),
                "weight_sparsity": 0.0,
            }
            return self._current_eval_result
        alpha_tensor = self._stack_panels(self.alpha_panels)
        target_panel = self._target_panel()
        z_struct = torch.stack([self._encode_struct_cached(expr) for expr in self.exprs], dim=0)
        z_alpha = torch.stack(
            [self._encode_alpha_cached(expr, panel) for expr, panel in zip(self.exprs, self.alpha_panels)],
            dim=0,
        )
        embeddings = torch.cat([z_alpha, z_struct], dim=-1)
        eval_result = self.evaluator.combine_precomputed(embeddings, alpha_tensor, target_panel)
        eval_result["z_alpha"] = z_alpha
        eval_result["z_struct"] = z_struct
        self._current_alpha_tensor = alpha_tensor
        self._current_struct_embeddings = z_struct
        self._current_alpha_embeddings = z_alpha
        self._current_eval_result = eval_result
        return eval_result

    def warm_start(self, exprs: Sequence[Expression]) -> None:
        for expr in exprs[: self.capacity]:
            try:
                panel = self._evaluate_expr(expr)
            except OutOfDataRangeError:
                continue
            self.exprs.append(expr)
            self.alpha_panels.append(panel)
        if self.exprs:
            current = self._refresh_cached_state()
            weights = np.asarray(current["weights"], dtype=np.float32)
            self.weights[: len(weights)] = weights
            self.best_metric = float(current["metric"])
            self.best_rankic = float(current["rankic"])

    def try_new_expr(self, expr: Expression, token_seq: Optional[List[str]] = None) -> Tuple[float, Dict]:
        try:
            candidate_panel = self._evaluate_expr(expr)
        except OutOfDataRangeError:
            return 0.0, {"out_of_data": True, "invalid": True}

        target_panel = self._target_panel()
        old_eval = self._evaluate_current()
        old_metric = float(old_eval["metric"])
        old_rankic = float(old_eval["rankic"])
        candidate_struct = self._encode_struct_cached(expr)
        candidate_alpha = self._encode_alpha_cached(expr, candidate_panel)
        if self._current_alpha_tensor is None or self._current_struct_embeddings is None or self._current_alpha_embeddings is None:
            self._refresh_cached_state()
        if self._current_alpha_tensor is None or self._current_struct_embeddings is None or self._current_alpha_embeddings is None:
            raise RuntimeError("ContextAlphaPool failed to initialize cached pool tensors.")
        new_panels_tensor = torch.cat([self._current_alpha_tensor, candidate_panel.unsqueeze(0)], dim=0)
        new_embeddings = torch.cat(
            [
                torch.cat([self._current_alpha_embeddings, candidate_alpha.unsqueeze(0)], dim=0),
                torch.cat([self._current_struct_embeddings, candidate_struct.unsqueeze(0)], dim=0),
            ],
            dim=-1,
        )
        new_eval = self.evaluator.combine_precomputed(new_embeddings, new_panels_tensor, target_panel)
        new_eval["z_alpha"] = torch.cat([self._current_alpha_embeddings, candidate_alpha.unsqueeze(0)], dim=0)
        new_eval["z_struct"] = torch.cat([self._current_struct_embeddings, candidate_struct.unsqueeze(0)], dim=0)
        re_value = float(new_eval["metric"]) - old_metric
        marginal_rankic_gain = float(new_eval["rankic"]) - old_rankic

        existing_panels = self._current_alpha_tensor if self._current_alpha_tensor is not None else torch.zeros((0,) + candidate_panel.shape)
        ri_func = self.evaluator.compute_ri_func(existing_panels, candidate_panel, target_panel)
        ri_struct, _ = self.evaluator.compute_ri_struct_from_embedding(candidate_struct, re_value=re_value)
        ri_reg = self.evaluator.compute_ri_reg(expr, candidate_panel)
        reward_total, reward_lambda = self.evaluator.compose_reward(
            step_idx=self.eval_cnt,
            re_value=re_value,
            ri_func=ri_func,
            ri_struct=ri_struct,
            ri_reg=ri_reg,
        )

        self.exprs.append(expr)
        self.alpha_panels.append(candidate_panel)
        self._current_alpha_tensor = new_panels_tensor
        self._current_struct_embeddings = new_eval["z_struct"]
        self._current_alpha_embeddings = new_eval["z_alpha"]
        self._current_eval_result = new_eval
        self._apply_eval_result(new_eval)
        self._prune_if_needed()
        cluster_update = self.evaluator.update_cluster_bank_from_embedding(candidate_struct, re_value)
        self.eval_cnt += 1

        info = {
            "re": float(re_value),
            "ri_func": float(ri_func),
            "ri_struct": float(ri_struct),
            "ri_reg": float(ri_reg),
            "reward_total": float(reward_total),
            "reward_pool": float(reward_total),
            "metric_before": float(old_metric),
            "metric_after": float(self.best_metric if self.size > 0 else 0.0),
            "marginal_rankic_gain": float(marginal_rankic_gain),
            "combiner_weight_sparsity": float(self.last_reward_info.get("combiner_weight_sparsity", 0.0)),
            "reward_lambda": float(reward_lambda),
            "cluster_id": int(cluster_update.cluster_id),
            "cluster_value": float(cluster_update.cluster_value),
            "cluster_coverage": float(cluster_update.cluster_coverage),
            "cluster_under_explore": float(cluster_update.under_explore),
            "cluster_is_new": bool(cluster_update.is_new),
            "cluster_distance": float(cluster_update.distance),
        }
        self.last_reward_info = info
        self.last_cluster_info = {
            "cluster_id": info["cluster_id"],
            "cluster_value": info["cluster_value"],
            "cluster_coverage": info["cluster_coverage"],
            "cluster_under_explore": info["cluster_under_explore"],
            "cluster_is_new": info["cluster_is_new"],
            "cluster_distance": info["cluster_distance"],
        }
        return reward_total, info

    def _apply_eval_result(self, eval_result: Dict[str, object]) -> None:
        weights = np.asarray(eval_result["weights"], dtype=np.float32)
        self.weights[:] = 0.0
        self.weights[: len(weights)] = weights
        self.best_metric = max(self.best_metric, float(eval_result["metric"]))
        self.best_rankic = max(self.best_rankic, float(eval_result["rankic"]))
        self.last_reward_info["combiner_weight_sparsity"] = float(eval_result.get("weight_sparsity", 0.0))

    def _prune_if_needed(self) -> None:
        if self.size <= self.capacity:
            return
        idx = int(np.argmin(np.abs(self.weights[: self.size])))
        self.exprs.pop(idx)
        self.alpha_panels.pop(idx)
        self.weights[:] = 0.0
        if self.exprs:
            eval_result = self._refresh_cached_state()
            weights = np.asarray(eval_result["weights"], dtype=np.float32)
            self.weights[: len(weights)] = weights
            self.last_reward_info["combiner_weight_sparsity"] = float(eval_result.get("weight_sparsity", 0.0))
        else:
            self._refresh_cached_state()

    def test_ensemble(self, calculator) -> Tuple[float, float]:
        if not self.exprs:
            return 0.0, 0.0
        if not isinstance(calculator, QLibStockDataCalculator):
            raise TypeError("ContextAlphaPool currently requires QLibStockDataCalculator for test_ensemble.")
        alpha_panels = [calculator._calc_alpha(expr).detach().cpu() for expr in self.exprs]
        eval_result = self.evaluator.evaluate_set(self.exprs, self._stack_panels(alpha_panels), calculator.target_value.detach().cpu())
        return float(eval_result["metric"]), float(eval_result["rankic"])
