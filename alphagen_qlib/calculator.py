import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
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
        self._alpha_cache_max_size = max(0, int(os.getenv("ALPHAGEN_ALPHA_CACHE_SIZE", "0")))
        self._alpha_cache: "OrderedDict[str, Tensor]" = OrderedDict()
        self._alpha_cache_hits = 0
        self._alpha_cache_misses = 0

        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data))

    def _alpha_cache_get(self, key: str) -> Optional[Tensor]:
        if self._alpha_cache_max_size <= 0:
            return None
        value = self._alpha_cache.get(key)
        if value is None:
            self._alpha_cache_misses += 1
            return None
        self._alpha_cache.move_to_end(key)
        self._alpha_cache_hits += 1
        return value

    def _alpha_cache_put(self, key: str, value: Tensor) -> None:
        if self._alpha_cache_max_size <= 0:
            return
        self._alpha_cache[key] = value
        self._alpha_cache.move_to_end(key)
        while len(self._alpha_cache) > self._alpha_cache_max_size:
            self._alpha_cache.popitem(last=False)

    def get_cache_stats(self) -> Dict[str, float]:
        total = self._alpha_cache_hits + self._alpha_cache_misses
        hit_rate = (self._alpha_cache_hits / total) if total > 0 else 0.0
        return {
            "alpha_cache_size": float(len(self._alpha_cache)),
            "alpha_cache_max_size": float(self._alpha_cache_max_size),
            "alpha_cache_hits": float(self._alpha_cache_hits),
            "alpha_cache_misses": float(self._alpha_cache_misses),
            "alpha_cache_hit_rate": float(hit_rate),
        }

    def _calc_alpha(self, expr: Expression) -> Tensor:
        key = str(expr)
        cached = self._alpha_cache_get(key)
        if cached is not None:
            return cached
        with torch.no_grad():
            value = normalize_by_day(expr.evaluate(self.data))
        self._alpha_cache_put(key, value)
        return value

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

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
