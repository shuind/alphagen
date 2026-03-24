from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import os
from torch import Tensor
import torch
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import StockData


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression]):
        self.data = data
        self._alpha_cache: "OrderedDict[str, Tensor]" = OrderedDict()
        self._alpha_cache_max_size: int = int(os.environ.get("ALPHAGEN_ALPHA_CACHE_SIZE", "128"))

        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data))

    def _calc_alpha(self, expr: Expression) -> Tensor:
        expr_key = str(expr)
        if self._alpha_cache_max_size <= 0:
            return normalize_by_day(expr.evaluate(self.data))
        cached = self._alpha_cache.get(expr_key)
        if cached is not None:
            self._alpha_cache.move_to_end(expr_key)
            return cached

        value = normalize_by_day(expr.evaluate(self.data))
        self._alpha_cache[expr_key] = value
        if len(self._alpha_cache) > self._alpha_cache_max_size:
            self._alpha_cache.popitem(last=False)
        return value

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) for i in range(n)]
        if n == 0:
            raise ValueError("exprs must not be empty")
        stacked = torch.stack(factors, dim=0)
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(n, 1, 1)
        return (stacked * weight_tensor).sum(dim=0)

    def calc_mutual_IC_batch(self, expr: Expression, others: List[Expression]) -> List[float]:
        if not others:
            return []
        value = self._calc_alpha(expr)
        other_values = torch.stack([self._calc_alpha(other) for other in others], dim=0)
        lhs = value.unsqueeze(0).expand_as(other_values)
        lhs_mean = lhs.mean(dim=2)
        rhs_mean = other_values.mean(dim=2)
        lhs_std = lhs.std(dim=2, unbiased=False)
        rhs_std = other_values.std(dim=2, unbiased=False)
        cov = (lhs * other_values).mean(dim=2) - lhs_mean * rhs_mean
        stdmul = lhs_std * rhs_std
        stdmul[(lhs_std < 1e-3) | (rhs_std < 1e-3)] = 1
        corr_by_day = cov / stdmul
        return corr_by_day.mean(dim=1).detach().cpu().tolist()

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value), self._calc_rIC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value)

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(ensemble_value, self.target_value)

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value), self._calc_rIC(ensemble_value, self.target_value)
