from itertools import count
import math
from typing import Dict, List, Optional, Tuple, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import Tensor
from alphagen.data.calculator import AlphaCalculator
from alphagen.config import MAX_EXPR_LENGTH

from alphagen.data.expression import (
    BinaryOperator,
    Constant,
    Corr,
    Cov,
    Div,
    Expression,
    Greater,
    Less,
    PairRollingOperator,
    RollingOperator,
    Sub,
    UnaryOperator,
)
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import masked_mean_std
from alphagen_qlib.stock_data import StockData


class AlphaPoolBase(metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        device: torch.device = torch.device('cpu')
    ):
        self.capacity = capacity
        self.calculator = calculator
        self.device = device

    @abstractmethod
    def to_dict(self) -> dict: ...

    @abstractmethod
    def try_new_expr(self, expr: Expression, token_seq: Optional[List[str]] = None) -> Tuple[float, Dict]: ...

    @abstractmethod
    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]: ...


class AlphaPool(AlphaPoolBase):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        reward_mode: str = "re",
        re_mode: str = "ensemble",
        lambda_ri: float = 0.0,
        ri_topk: int = 5,
        ri_struct_topk: int = 5,
        ri_reg_l0: Optional[float] = None,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__(capacity, calculator, device)

        self.size: int = 0
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.expr_tokens: List[Optional[List[str]]] = [None for _ in range(capacity + 1)]
        self.expr_bigrams: List[Optional[Set[Tuple[str, str]]]] = [None for _ in range(capacity + 1)]
        self.expr_keys: List[Optional[str]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self.mutual_ics: np.ndarray = np.identity(capacity + 1)
        self.weights: np.ndarray = np.zeros(capacity + 1)
        self.best_ic_ret: float = -1.
        self._single_ic_cache: Dict[str, float] = {}
        self._mutual_ic_cache: Dict[Tuple[str, str], float] = {}

        self.ic_lower_bound = ic_lower_bound or -1.
        self.l1_alpha = l1_alpha
        self.reward_mode = reward_mode
        self.re_mode = re_mode
        self.lambda_ri = lambda_ri
        self.ri_topk = ri_topk
        self.ri_struct_topk = ri_struct_topk
        self.ri_reg_l0 = float(ri_reg_l0) if ri_reg_l0 is not None else float(int(0.7 * MAX_EXPR_LENGTH))

        self.eval_cnt = 0

    @property
    def state(self) -> dict:
        return {
            "exprs": list(self.exprs[:self.size]),
            "ics_ret": list(self.single_ics[:self.size]),
            "weights": list(self.weights[:self.size]),
            "best_ic_ret": self.best_ic_ret
        }

    def to_dict(self) -> dict:
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights[:self.size])
        }

    def _expr_key(self, expr: Expression) -> str:
        return str(expr)

    def _mutual_key(self, lhs_key: str, rhs_key: str) -> Tuple[str, str]:
        return (lhs_key, rhs_key) if lhs_key <= rhs_key else (rhs_key, lhs_key)

    def try_new_expr(self, expr: Expression, token_seq: Optional[List[str]] = None) -> Tuple[float, Dict]:
        if not self._semantic_validate(expr):
            info = {
                "re": 0.0,
                "ri_func": 0.0,
                "ri_struct": 0.0,
                "ri_reg": 0.0,
                "reward_total": 0.0,
                "reward_pool": 0.0,
                "ic_ensemble": 0.0,
                "increment": 0.0,
                "invalid": True,
            }
            return 0.0, info
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
            info = {
                "re": 0.0,
                "ri_func": 0.0,
                "ri_struct": 0.0,
                "ri_reg": 0.0,
                "reward_total": 0.0,
                "reward_pool": 0.0,
                "ic_ensemble": 0.0,
                "increment": 0.0,
                "invalid": True,
            }
            return 0.0, info

        ri_func = self._calc_ri_func(ic_mut) if self._use_ri_func else 0.0
        ri_struct = self._calc_ri_struct(token_seq) if self._use_ri_struct else 0.0
        ri_reg = self._calc_ri_reg(token_seq) if self._use_ri_reg else 0.0

        prev_best_ic_ret = self.best_ic_ret
        self._add_factor(expr, ic_ret, ic_mut, token_seq)
        if self.size > 1:
            new_weights = self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=500)
            worst_idx = np.argmin(np.abs(new_weights))
            if worst_idx != self.capacity:
                self.weights[:self.size] = new_weights
            self._pop()

        new_ic_ret = self.evaluate_ensemble()
        increment = new_ic_ret - prev_best_ic_ret
        if increment > 0:
            self.best_ic_ret = new_ic_ret
        self.eval_cnt += 1
        re = self._compose_re(new_ic_ret, increment)
        reward_total = self._compose_reward(re, ri_func, ri_struct, ri_reg)
        info = {
            "re": float(re),
            "ri_func": float(ri_func),
            "ri_struct": float(ri_struct),
            "ri_reg": float(ri_reg),
            "reward_total": float(reward_total),
            "reward_pool": float(reward_total),
            "ic_ensemble": float(new_ic_ret),
            "increment": float(increment),
            "ic_single": float(ic_ret),
            "re_mode": self.re_mode,
        }
        return reward_total, info

    def force_load_exprs(self, exprs: List[Expression]) -> None:
        for expr in exprs:
            ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, ic_ret, ic_mut, None)
            assert self.size <= self.capacity
        self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=500)

    def _optimize(self, alpha: float, lr: float, n_iter: int) -> np.ndarray:
        if math.isclose(alpha, 0.): # no L1 regularization
            return self._optimize_lstsq() # very fast

        ics_ret = torch.from_numpy(self.single_ics[:self.size]).to(self.device)
        ics_mut = torch.from_numpy(self.mutual_ics[:self.size, :self.size]).to(self.device)
        weights = torch.from_numpy(self.weights[:self.size]).to(self.device).requires_grad_()
        optim = torch.optim.Adam([weights], lr=lr)

        loss_ic_min = 1e9 + 7  # An arbitrary big value
        best_weights = weights.cpu().detach().numpy()
        iter_cnt = 0
        for it in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()

            loss_l1 = torch.norm(weights, p=1)  # type: ignore
            loss = loss_ic + alpha * loss_l1

            optim.zero_grad()
            loss.backward()
            optim.step()

            if loss_ic_min - loss_ic_curr > 1e-6:
                iter_cnt = 0
            else:
                iter_cnt += 1

            if loss_ic_curr < loss_ic_min:
                best_weights = weights.cpu().detach().numpy()
                loss_ic_min = loss_ic_curr

            if iter_cnt >= n_iter or it >= 10000:
                break

        return best_weights

    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(self.mutual_ics[:self.size, :self.size],self.single_ics[:self.size])[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights[:self.size]

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        ic, rank_ic = calculator.calc_pool_all_ret(self.exprs[:self.size], self.weights[:self.size])
        return ic, rank_ic

    def evaluate_ensemble(self) -> float:
        ic = self.calculator.calc_pool_IC_ret(self.exprs[:self.size], self.weights[:self.size])
        return ic

    @property
    def _under_thres_alpha(self) -> bool:
        if self.ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self.ic_lower_bound

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]]]:
        if not self._semantic_validate(expr):
            return 0.0, None
        expr_key = self._expr_key(expr)
        if expr_key in self._single_ic_cache:
            single_ic = self._single_ic_cache[expr_key]
        else:
            single_ic = self.calculator.calc_single_IC_ret(expr)
            self._single_ic_cache[expr_key] = single_ic
        if not self._under_thres_alpha and single_ic < self.ic_lower_bound:
            return single_ic, None

        mutual_ics: List[Optional[float]] = [None for _ in range(self.size)]
        batch_exprs: List[Expression] = []
        batch_keys: List[Tuple[str, str]] = []
        batch_indices: List[int] = []
        for i in range(self.size):
            existing_expr = self.exprs[i]
            existing_key = self.expr_keys[i]
            assert existing_expr is not None and existing_key is not None
            cache_key = self._mutual_key(expr_key, existing_key)
            if cache_key in self._mutual_ic_cache:
                mutual_ic = self._mutual_ic_cache[cache_key]
                if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                    return single_ic, None
                mutual_ics[i] = mutual_ic
            else:
                batch_exprs.append(existing_expr)
                batch_keys.append(cache_key)
                batch_indices.append(i)

        if batch_exprs:
            if hasattr(self.calculator, "calc_mutual_IC_batch"):
                batch_values = self.calculator.calc_mutual_IC_batch(expr, batch_exprs)
            else:
                batch_values = [self.calculator.calc_mutual_IC(expr, other) for other in batch_exprs]
            for idx, cache_key, mutual_ic in zip(batch_indices, batch_keys, batch_values):
                self._mutual_ic_cache[cache_key] = mutual_ic
                if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                    return single_ic, None
                mutual_ics[idx] = mutual_ic

        return single_ic, [float(v) for v in mutual_ics]

    def _semantic_validate(self, expr: Expression) -> bool:
        for node in self._iter_expr_nodes(expr):
            if isinstance(node, (Corr, Cov)) and self._expr_equal(node._lhs, node._rhs):
                return False
            if isinstance(node, (Sub, Div)) and self._expr_equal(node._lhs, node._rhs):
                return False
            if isinstance(node, (Less, Greater)):
                if self._is_bool_chain(node._lhs) and not self._is_bool_chain(node._rhs):
                    return False
                if self._is_bool_chain(node._rhs) and not self._is_bool_chain(node._lhs):
                    return False
                if self._is_unrealistic_negative_compare(node._lhs, node._rhs):
                    return False
                if self._is_unrealistic_negative_compare(node._rhs, node._lhs):
                    return False
        return True

    def _iter_expr_nodes(self, expr: Expression):
        yield expr
        if isinstance(expr, UnaryOperator):
            yield from self._iter_expr_nodes(expr._operand)
        elif isinstance(expr, BinaryOperator):
            yield from self._iter_expr_nodes(expr._lhs)
            yield from self._iter_expr_nodes(expr._rhs)
        elif isinstance(expr, RollingOperator):
            yield from self._iter_expr_nodes(expr._operand)
        elif isinstance(expr, PairRollingOperator):
            yield from self._iter_expr_nodes(expr._lhs)
            yield from self._iter_expr_nodes(expr._rhs)

    def _expr_equal(self, lhs: Expression, rhs: Expression) -> bool:
        return str(lhs) == str(rhs)

    def _is_bool_chain(self, expr: Expression) -> bool:
        return isinstance(expr, (Less, Greater))

    def _is_unrealistic_negative_compare(self, lhs: Expression, rhs: Expression) -> bool:
        if not isinstance(rhs, Constant):
            return False
        if rhs._value >= 0:
            return False
        lhs_repr = str(lhs).lower()
        if "$volume" in lhs_repr or "$open" in lhs_repr or "$close" in lhs_repr or "$high" in lhs_repr or "$low" in lhs_repr or "$vwap" in lhs_repr:
            return True
        return False

    def _add_factor(
        self,
        expr: Expression,
        ic_ret: float,
        ic_mut: List[float],
        token_seq: Optional[List[str]]
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.expr_keys[n] = self._expr_key(expr)
        self.expr_tokens[n] = token_seq
        if token_seq is not None and len(token_seq) >= 2:
            self.expr_bigrams[n] = self._token_bigrams(token_seq)
        else:
            self.expr_bigrams[n] = None
        self.single_ics[n] = ic_ret
        for i in range(n):
            self.mutual_ics[i][n] = self.mutual_ics[n][i] = ic_mut[i]
        self.weights[n] = ic_ret  # An arbitrary init value
        self.size += 1

    def _pop(self) -> None:
        if self.size <= self.capacity:
            return
        idx = np.argmin(np.abs(self.weights))
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity

    def _swap_idx(self, i, j) -> None:
        if i == j:
            return
        self.exprs[i], self.exprs[j] = self.exprs[j], self.exprs[i]
        self.expr_keys[i], self.expr_keys[j] = self.expr_keys[j], self.expr_keys[i]
        self.expr_tokens[i], self.expr_tokens[j] = self.expr_tokens[j], self.expr_tokens[i]
        self.expr_bigrams[i], self.expr_bigrams[j] = self.expr_bigrams[j], self.expr_bigrams[i]
        self.single_ics[i], self.single_ics[j] = self.single_ics[j], self.single_ics[i]
        self.mutual_ics[:, [i, j]] = self.mutual_ics[:, [j, i]]
        self.mutual_ics[[i, j], :] = self.mutual_ics[[j, i], :]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]

    @property
    def _use_ri_func(self) -> bool:
        return "func" in self.reward_mode or "all" in self.reward_mode

    @property
    def _use_ri_struct(self) -> bool:
        return "struct" in self.reward_mode or "all" in self.reward_mode

    @property
    def _use_ri_reg(self) -> bool:
        return "reg" in self.reward_mode or "all" in self.reward_mode

    def _compose_re(self, ic_ensemble: float, increment: float) -> float:
        if self.re_mode == "delta_best":
            return increment
        return ic_ensemble

    def _compose_reward(
        self,
        re: float,
        ri_func: float,
        ri_struct: float,
        ri_reg: float
    ) -> float:
        if self.reward_mode == "re":
            return re
        ri_sum = 0.0
        if self._use_ri_func:
            ri_sum += ri_func
        if self._use_ri_struct:
            ri_sum += ri_struct
        if self._use_ri_reg:
            ri_sum += ri_reg
        return re + self.lambda_ri * ri_sum

    def _calc_ri_func(self, ic_mut: List[float]) -> float:
        if not ic_mut:
            return 0.0
        valid = [abs(x) for x in ic_mut if not np.isnan(x)]
        if not valid:
            return 0.0
        k = max(1, min(self.ri_topk, len(valid)))
        topk = sorted(valid)[-k:]
        return -float(np.mean(topk))

    def _calc_ri_struct(self, token_seq: Optional[List[str]]) -> float:
        if token_seq is None or len(token_seq) < 2 or self.size == 0:
            return 0.0
        new_bigrams = self._token_bigrams(token_seq)
        if not new_bigrams:
            return 0.0
        max_jaccard = 0.0
        pool_size = self.size
        if pool_size <= 0:
            return 0.0
        topk = max(1, min(self.ri_struct_topk, pool_size))
        weight_scores = np.abs(self.weights[:pool_size])
        idxs = np.argsort(weight_scores)[-topk:]
        for i in idxs:
            old_bigrams = self.expr_bigrams[i]
            if not old_bigrams:
                continue
            inter = len(new_bigrams & old_bigrams)
            union = len(new_bigrams | old_bigrams)
            if union == 0:
                continue
            max_jaccard = max(max_jaccard, inter / union)
        return 1.0 - max_jaccard

    def _calc_ri_reg(self, token_seq: Optional[List[str]]) -> float:
        if token_seq is None:
            return 0.0
        length = len(token_seq)
        if length <= 0:
            return 0.0
        l0 = max(1.0, float(self.ri_reg_l0))
        return -max(0.0, (length - l0) / l0)

    @staticmethod
    def _token_bigrams(token_seq: List[str]) -> Set[Tuple[str, str]]:
        return set(zip(token_seq[:-1], token_seq[1:]))
