from itertools import count
import math
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import Tensor
from alphagen.data.calculator import AlphaCalculator
from alphagen.config import MAX_EXPR_LENGTH

from alphagen.data.expression import (
    BinaryOperator,
    Corr,
    Cov,
    Div,
    Expression,
    Greater,
    Less,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)
from alphagen.models.alpha_cluster import cosine_similarity, extract_ast_feature_vector
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
        ri_func_weight: float = 1.0,
        ri_struct_weight: float = 1.0,
        ri_reg_weight: float = 1.0,
        ri_schedule_decay: float = 0.0,
        ri_struct_value_bonus: float = 0.1,
        ri_struct_underexplore_power: float = 1.0,
        ri_func_metric: str = "rankic",
        ri_func_topk: int = 8,
        ri_func_sample_size: int = 128,
        ri_func_rankic_on_cpu: bool = True,
        ri_admission_gate: bool = False,
        ri_func_backend: str = "auto",
        ri_struct_backend: str = "token_bigram",
        ri_corr_threshold: float = 0.8,
        ri_ast_similarity_threshold: float = 0.9,
        ri_corr_value_bonus: float = 0.1,
        ri_corr_underexplore_power: float = 1.0,
        ri_turnover_weight: float = 0.0,
        ri_turnover_topk: int = 30,
        ri_turnover_baseline: float = 0.5,
        ri_turnover_step_stride: int = 5,
        profile_timing: bool = True,
        optimize_every: int = 2,
        optimize_n_iter: int = 256,
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
        self.expr_ast_features: List[Optional[np.ndarray]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self.mutual_ics: np.ndarray = np.identity(capacity + 1)
        self.weights: np.ndarray = np.zeros(capacity + 1)
        self.best_ic_ret: float = -1.

        self.ic_lower_bound = ic_lower_bound or -1.
        self.l1_alpha = l1_alpha
        self.reward_mode = reward_mode
        self.re_mode = re_mode
        self.lambda_ri = lambda_ri
        self.ri_func_weight = float(ri_func_weight)
        self.ri_struct_weight = float(ri_struct_weight)
        self.ri_reg_weight = float(ri_reg_weight)
        self.ri_schedule_decay = float(ri_schedule_decay)
        self.ri_struct_value_bonus = float(ri_struct_value_bonus)
        self.ri_struct_underexplore_power = float(ri_struct_underexplore_power)
        self.ri_func_metric = str(ri_func_metric)
        self.ri_func_topk = max(1, int(ri_func_topk))
        self.ri_func_sample_size = max(0, int(ri_func_sample_size))
        self.ri_func_rankic_on_cpu = bool(ri_func_rankic_on_cpu)
        self.ri_admission_gate = bool(ri_admission_gate)
        self.ri_func_backend = self._resolve_ri_func_backend(ri_func_backend)
        self.ri_struct_backend = str(ri_struct_backend or "token_bigram")
        self.ri_corr_threshold = float(ri_corr_threshold)
        self.ri_ast_similarity_threshold = float(ri_ast_similarity_threshold)
        self.ri_corr_value_bonus = float(ri_corr_value_bonus)
        self.ri_corr_underexplore_power = float(ri_corr_underexplore_power)
        self.ri_turnover_weight = float(ri_turnover_weight)
        self.ri_turnover_topk = max(1, int(ri_turnover_topk))
        self.ri_turnover_baseline = float(ri_turnover_baseline)
        self.ri_turnover_step_stride = max(1, int(ri_turnover_step_stride))
        self.profile_timing = bool(profile_timing)
        self.optimize_every = max(1, int(optimize_every))
        self.optimize_n_iter = max(1, int(optimize_n_iter))
        self.ri_topk = ri_topk
        self.ri_struct_topk = ri_struct_topk
        self.ri_reg_l0 = float(ri_reg_l0) if ri_reg_l0 is not None else float(int(0.7 * MAX_EXPR_LENGTH))

        self.eval_cnt = 0
        self._optimize_eval_counter = 0
        self.last_reward_info: Dict[str, Any] = {}
        self._ri_func_timing: Dict[str, Any] = {}
        self._structure_clusters: List[Dict[str, Any]] = []
        self._ast_structure_clusters: List[Dict[str, Any]] = []
        self._corr_clusters: List[Dict[str, Any]] = []
        self._timing_total: Dict[str, float] = {}
        self._timing_count: Dict[str, int] = {}
        self._single_ic_cache_size = max(0, int(os.getenv("ALPHAGEN_SINGLE_IC_CACHE_SIZE", "8192")))
        self._single_ic_cache: "OrderedDict[str, float]" = OrderedDict()
        self._single_ic_cache_hits = 0
        self._single_ic_cache_misses = 0
        self._mutual_ic_cache_size = max(0, int(os.getenv("ALPHAGEN_MUTUAL_IC_CACHE_SIZE", "131072")))
        self._mutual_ic_cache: "OrderedDict[Tuple[str, str], float]" = OrderedDict()
        self._mutual_ic_cache_hits = 0
        self._mutual_ic_cache_misses = 0

    def _record_timing(self, name: str, elapsed_sec: float) -> None:
        if not self.profile_timing:
            return
        self._timing_total[name] = float(self._timing_total.get(name, 0.0) + elapsed_sec)
        self._timing_count[name] = int(self._timing_count.get(name, 0) + 1)

    def _lru_get(self, cache: "OrderedDict", key: Any) -> Any:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value

    def _lru_put(self, cache: "OrderedDict", key: Any, value: Any, max_size: int) -> None:
        if max_size <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _get_single_ic_cached(self, expr: Expression) -> float:
        key = self._expr_key(expr)
        cached = self._lru_get(self._single_ic_cache, key)
        if cached is not None:
            self._single_ic_cache_hits += 1
            return float(cached)
        self._single_ic_cache_misses += 1
        value = float(self.calculator.calc_single_IC_ret(expr))
        self._lru_put(self._single_ic_cache, key, value, self._single_ic_cache_size)
        return value

    def _get_mutual_ic_cached(self, lhs: Expression, rhs: Expression) -> float:
        lhs_key = self._expr_key(lhs)
        rhs_key = self._expr_key(rhs)
        key = self._mutual_key(lhs_key, rhs_key)
        cached = self._lru_get(self._mutual_ic_cache, key)
        if cached is not None:
            self._mutual_ic_cache_hits += 1
            return float(cached)
        self._mutual_ic_cache_misses += 1
        value = float(self.calculator.calc_mutual_IC(lhs, rhs))
        self._lru_put(self._mutual_ic_cache, key, value, self._mutual_ic_cache_size)
        return value

    def _cache_snapshot(self) -> Dict[str, float]:
        single_total = self._single_ic_cache_hits + self._single_ic_cache_misses
        mutual_total = self._mutual_ic_cache_hits + self._mutual_ic_cache_misses
        info: Dict[str, float] = {
            "single_ic_cache_size": float(len(self._single_ic_cache)),
            "single_ic_cache_max_size": float(self._single_ic_cache_size),
            "single_ic_cache_hits": float(self._single_ic_cache_hits),
            "single_ic_cache_misses": float(self._single_ic_cache_misses),
            "single_ic_cache_hit_rate": float(self._single_ic_cache_hits / single_total) if single_total > 0 else 0.0,
            "mutual_ic_cache_size": float(len(self._mutual_ic_cache)),
            "mutual_ic_cache_max_size": float(self._mutual_ic_cache_size),
            "mutual_ic_cache_hits": float(self._mutual_ic_cache_hits),
            "mutual_ic_cache_misses": float(self._mutual_ic_cache_misses),
            "mutual_ic_cache_hit_rate": float(self._mutual_ic_cache_hits / mutual_total) if mutual_total > 0 else 0.0,
        }
        calc_stats = getattr(self.calculator, "get_cache_stats", None)
        if callable(calc_stats):
            try:
                raw = calc_stats()
                for k, v in raw.items():
                    info[k] = float(v)
            except Exception:
                pass
        return info

    def profile_snapshot(self) -> Dict[str, Any]:
        totals = dict(self._timing_total)
        counts = dict(self._timing_count)
        tracked_total_sec = float(sum(totals.values()))
        avg_ms = {
            f"{k}_avg_ms": (
                float((totals[k] / max(1, counts.get(k, 0))) * 1000.0)
                if counts.get(k, 0) > 0 else 0.0
            )
            for k in totals.keys()
        }
        return {
            "timing_totals_sec": totals,
            "timing_counts": counts,
            "timing_tracked_total_sec": tracked_total_sec,
            "timing_avg_ms": avg_ms,
            "eval_cnt": int(self.eval_cnt),
            "profile_timing_enabled": bool(self.profile_timing),
        }

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
            "weights": list(self.weights[:self.size]),
            "last_reward_info": self.last_reward_info,
            "cluster_bank_summary": self.cluster_bank_summary,
            "corr_cluster_bank_summary": self.corr_cluster_bank_summary,
            "reward_backends": {
                "ri_func_backend": self.ri_func_backend,
                "ri_struct_backend": self.ri_struct_backend,
                "ri_corr_threshold": self.ri_corr_threshold,
                "ri_ast_similarity_threshold": self.ri_ast_similarity_threshold,
            },
        }

    @property
    def cluster_bank_summary(self) -> List[Dict[str, Any]]:
        if self.ri_struct_backend == "ast_cluster":
            return self.ast_cluster_bank_summary
        return self.token_cluster_bank_summary

    @property
    def token_cluster_bank_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "cluster_id": int(cluster["cluster_id"]),
                "cluster_count": int(cluster["count"]),
                "cluster_mean_re": float(cluster["mean_re"]),
                "cluster_positive_re_rate": float(cluster["positive_re_rate"]),
                "last_seen_eval": int(cluster["last_seen_eval"]),
                "prototype_size": int(len(cluster.get("prototype", []))),
            }
            for cluster in self._structure_clusters
        ]

    @property
    def ast_cluster_bank_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "cluster_id": int(cluster["cluster_id"]),
                "cluster_count": int(cluster["count"]),
                "cluster_mean_re": float(cluster["mean_re"]),
                "cluster_positive_re_rate": float(cluster["positive_re_rate"]),
                "last_seen_eval": int(cluster["last_seen_eval"]),
                "prototype_expr_key": str(cluster.get("prototype_expr_key", "")),
            }
            for cluster in self._ast_structure_clusters
        ]

    @property
    def corr_cluster_bank_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "cluster_id": int(cluster["cluster_id"]),
                "cluster_count": int(cluster["count"]),
                "cluster_mean_re": float(cluster["mean_re"]),
                "cluster_positive_re_rate": float(cluster["positive_re_rate"]),
                "last_seen_eval": int(cluster["last_seen_eval"]),
                "representative_expr_key": str(cluster.get("representative_expr_key", "")),
            }
            for cluster in self._corr_clusters
        ]

    def _expr_key(self, expr: Expression) -> str:
        return str(expr)

    def _mutual_key(self, lhs_key: str, rhs_key: str) -> Tuple[str, str]:
        return (lhs_key, rhs_key) if lhs_key <= rhs_key else (rhs_key, lhs_key)

    def _resolve_ri_func_backend(self, backend: str) -> str:
        backend = str(backend or "auto")
        if backend == "auto":
            return "residual" if self.reward_mode.startswith("re_v2") else "mutual_ic"
        return backend
    def try_new_expr(self, expr: Expression, token_seq: Optional[List[str]] = None) -> Tuple[float, Dict]:
        t_try0 = time.perf_counter()
        t0 = time.perf_counter()
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
        self._record_timing("calc_ics_sec", time.perf_counter() - t0)
        if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
            self._record_timing("invalid_expr_sec", time.perf_counter() - t_try0)
            self._record_timing("try_new_expr_total_sec", time.perf_counter() - t_try0)
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

        expr_ast_feature = extract_ast_feature_vector(expr)
        corr_cluster_pending: Optional[Dict[str, Any]] = None
        t0 = time.perf_counter()
        ri_func = 0.0
        ri_func_info: Dict[str, Any] = {}
        if self._use_ri_func:
            ri_func, ri_func_info, corr_cluster_pending = self._calc_ri_func_dispatch(expr, ic_mut, expr_ast_feature)
        self._record_timing("ri_func_sec", time.perf_counter() - t0)
        t0 = time.perf_counter()
        ri_reg = self._calc_ri_reg_v2(expr, token_seq) if self._use_v2 and self._use_ri_reg else (
            self._calc_ri_reg(token_seq) if self._use_ri_reg else 0.0
        )
        self._record_timing("ri_reg_sec", time.perf_counter() - t0)

        prev_best_ic_ret = self.best_ic_ret
        t0 = time.perf_counter()
        self._add_factor(expr, ic_ret, ic_mut, token_seq, expr_ast_feature)
        self._record_timing("add_factor_sec", time.perf_counter() - t0)
        self._optimize_eval_counter += 1
        optimize_executed = False
        if self.size > 1:
            if self._optimize_eval_counter % self.optimize_every == 0:
                t0 = time.perf_counter()
                new_weights = self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=self.optimize_n_iter)
                self._record_timing("optimize_sec", time.perf_counter() - t0)
                worst_idx = np.argmin(np.abs(new_weights))
                if worst_idx != self.capacity:
                    self.weights[:self.size] = new_weights
                optimize_executed = True
            t0 = time.perf_counter()
            self._pop()
            self._record_timing("pop_sec", time.perf_counter() - t0)

        t0 = time.perf_counter()
        new_ic_ret = self.evaluate_ensemble()
        self._record_timing("evaluate_ensemble_sec", time.perf_counter() - t0)
        turnover_mean = math.nan
        turnover_penalty = 0.0
        if self._use_ri_reg and self.ri_turnover_weight > 0.0:
            turnover_mean, turnover_penalty = self._calc_pool_turnover_penalty()
            ri_reg += self.ri_turnover_weight * turnover_penalty
        increment = new_ic_ret - prev_best_ic_ret
        if increment > 0:
            self.best_ic_ret = new_ic_ret
        self.eval_cnt += 1
        re = self._compose_re(new_ic_ret, increment)
        cluster_info: Dict[str, Any] = {}
        ri_struct = 0.0
        if self._use_ri_struct:
            t0 = time.perf_counter()
            ri_struct, cluster_info = self._calc_ri_struct_dispatch(expr, token_seq, re, expr_ast_feature)
            self._record_timing("ri_struct_sec", time.perf_counter() - t0)
        corr_cluster_info: Dict[str, Any] = {}
        if corr_cluster_pending is not None:
            corr_cluster_info = self._finalize_corr_cluster(corr_cluster_pending, expr, re)
        t0 = time.perf_counter()
        reward_total, reward_lambda_t = self._compose_reward(re, ri_func, ri_struct, ri_reg)
        self._record_timing("compose_reward_sec", time.perf_counter() - t0)
        self._record_timing("try_new_expr_total_sec", time.perf_counter() - t_try0)
        info = {
            "re": float(re),
            "ri_func": float(ri_func),
            "ri_struct": float(ri_struct),
            "ri_reg": float(ri_reg),
            "turnover_mean": float(turnover_mean) if math.isfinite(turnover_mean) else math.nan,
            "turnover_penalty": float(turnover_penalty),
            "turnover_weight": float(self.ri_turnover_weight),
            "turnover_topk": int(self.ri_turnover_topk),
            "turnover_baseline": float(self.ri_turnover_baseline),
            "turnover_step_stride": int(self.ri_turnover_step_stride),
            "reward_total": float(reward_total),
            "reward_pool": float(reward_total),
            "ic_ensemble": float(new_ic_ret),
            "increment": float(increment),
            "ic_single": float(ic_ret),
            "re_mode": self.re_mode,
            "reward_lambda_t": float(reward_lambda_t),
            "ri_func_backend": self.ri_func_backend,
            "ri_struct_backend": self.ri_struct_backend,
            "optimize_executed": bool(optimize_executed),
            "optimize_every": int(self.optimize_every),
            "optimize_n_iter": int(self.optimize_n_iter),
            "admission_gate_enabled": bool(self.ri_admission_gate and self._use_v2),
            "admission_gate_passed": bool((not self.ri_admission_gate) or (re > 0)),
        }
        if isinstance(getattr(self, "_ri_func_timing", None), dict):
            info.update(getattr(self, "_ri_func_timing"))
        if ri_func_info:
            info.update(ri_func_info)
        info.update(self._cache_snapshot())
        if self.profile_timing:
            prof = self.profile_snapshot()
            info.update(
                {
                    "profile_try_total_sec": prof["timing_totals_sec"].get("try_new_expr_total_sec", 0.0),
                    "profile_calc_ics_sec": prof["timing_totals_sec"].get("calc_ics_sec", 0.0),
                    "profile_ri_func_sec": prof["timing_totals_sec"].get("ri_func_sec", 0.0),
                    "profile_ri_struct_sec": prof["timing_totals_sec"].get("ri_struct_sec", 0.0),
                    "profile_ri_reg_sec": prof["timing_totals_sec"].get("ri_reg_sec", 0.0),
                    "profile_optimize_sec": prof["timing_totals_sec"].get("optimize_sec", 0.0),
                    "profile_eval_ensemble_sec": prof["timing_totals_sec"].get("evaluate_ensemble_sec", 0.0),
                }
            )
        if cluster_info:
            info.update(cluster_info)
        if corr_cluster_info:
            info.update(corr_cluster_info)
        self.last_reward_info = info
        return reward_total, info

    def force_load_exprs(self, exprs: List[Expression]) -> None:
        for expr in exprs:
            ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, ic_ret, ic_mut, None, extract_ast_feature_vector(expr))
            assert self.size <= self.capacity
        self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=self.optimize_n_iter)

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
            # Equivalent quadratic form, but avoids allocating outer(weights, weights) each iteration.
            ret_ic_sum = torch.dot(weights, ics_ret)
            mut_ic_sum = torch.dot(weights, torch.mv(ics_mut, weights))
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
            a = self.mutual_ics[:self.size, :self.size]
            b = self.single_ics[:self.size]
            try:
                return np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(a, b, rcond=None)[0]
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
        single_ic = self._get_single_ic_cached(expr)
        if not self._under_thres_alpha and single_ic < self.ic_lower_bound:
            return single_ic, None

        mutual_ics = []
        for i in range(self.size):
            prev_expr = self.exprs[i]
            if prev_expr is None:
                raise RuntimeError(f"pool expr[{i}] is None while size={self.size}")
            mutual_ic = self._get_mutual_ic_cached(expr, prev_expr)
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics

    def _add_factor(
        self,
        expr: Expression,
        ic_ret: float,
        ic_mut: List[float],
        token_seq: Optional[List[str]],
        ast_feature: Optional[np.ndarray]
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.expr_tokens[n] = token_seq
        if token_seq is not None and len(token_seq) >= 2:
            self.expr_bigrams[n] = self._token_bigrams(token_seq)
        else:
            self.expr_bigrams[n] = None
        self.expr_ast_features[n] = None if ast_feature is None else np.asarray(ast_feature, dtype=np.float32)
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
        self.expr_tokens[i], self.expr_tokens[j] = self.expr_tokens[j], self.expr_tokens[i]
        self.expr_bigrams[i], self.expr_bigrams[j] = self.expr_bigrams[j], self.expr_bigrams[i]
        self.expr_ast_features[i], self.expr_ast_features[j] = self.expr_ast_features[j], self.expr_ast_features[i]
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

    @property
    def _use_v2(self) -> bool:
        return self.reward_mode.startswith("re_v2")

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
    ) -> Tuple[float, float]:
        reward_lambda = self.lambda_ri
        if self._use_v2:
            reward_lambda = self.lambda_ri / (1.0 + self.ri_schedule_decay * max(self.eval_cnt, 0))
        if self.reward_mode == "re":
            return re, reward_lambda
        if self.reward_mode == "re_v2":
            return re, reward_lambda
        ri_sum = 0.0
        if self._use_ri_func:
            ri_sum += self.ri_func_weight * ri_func
        if self._use_ri_struct:
            ri_sum += self.ri_struct_weight * ri_struct
        if self._use_ri_reg:
            ri_sum += self.ri_reg_weight * ri_reg
        return re + reward_lambda * ri_sum, reward_lambda

    def _calc_ri_func(self, ic_mut: List[float]) -> float:
        if not ic_mut:
            return 0.0
        valid = [abs(x) for x in ic_mut if not np.isnan(x)]
        if not valid:
            return 0.0
        k = max(1, min(self.ri_topk, len(valid)))
        topk = sorted(valid)[-k:]
        return -float(np.mean(topk))

    def _calc_ri_func_v2(self, expr: Expression) -> float:
        t0 = time.perf_counter()
        candidate_value = self.calculator._calc_alpha(expr)
        t1 = time.perf_counter()
        target_value = getattr(self.calculator, "target_value", None)
        if target_value is None:
            self._ri_func_timing = {"ri_func_total_ms": 0.0}
            return 0.0
        if self.size <= 0:
            metric = batch_spearmanr(candidate_value, target_value) if self.ri_func_metric == "rankic" else batch_pearsonr(candidate_value, target_value)
            t2 = time.perf_counter()
            self._ri_func_timing = {
                "ri_func_eval_ms": (t1 - t0) * 1000.0,
                "ri_func_stack_ms": 0.0,
                "ri_func_lstsq_ms": 0.0,
                "ri_func_metric_ms": (t2 - t1) * 1000.0,
                "ri_func_total_ms": (t2 - t0) * 1000.0,
                "ri_func_used_k": 0,
                "ri_func_used_sample": 0,
            }
            return float(metric.mean().item())

        used_k = min(self.size, self.ri_func_topk)
        abs_w = np.abs(self.weights[:self.size])
        if np.all(np.isnan(abs_w)):
            idxs = np.arange(self.size)
        else:
            idxs = np.argsort(np.nan_to_num(abs_w, nan=0.0))[-used_k:]
        sel_exprs = [self.exprs[int(i)] for i in idxs]
        pool_values = torch.stack([self.calculator._calc_alpha(e) for e in sel_exprs], dim=0)
        t2 = time.perf_counter()
        x = pool_values.permute(1, 2, 0).reshape(-1, used_k).float()
        y = candidate_value.reshape(-1).float()
        used_sample = 0
        if self.ri_func_sample_size > 0 and x.shape[0] > self.ri_func_sample_size:
            sample_idx = torch.randperm(x.shape[0], device=x.device)[: self.ri_func_sample_size]
            x = x[sample_idx]
            y = y[sample_idx]
            used_sample = int(self.ri_func_sample_size)
        else:
            used_sample = int(x.shape[0])
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        beta = torch.linalg.lstsq(x, y).solution
        t3 = time.perf_counter()
        resid_flat = y - x @ beta
        # For sampled path, metric is computed on sampled vectors to reduce memory footprint.
        if self.ri_func_metric == "rankic":
            if self.ri_func_rankic_on_cpu:
                resid_metric = resid_flat.detach().cpu().view(1, -1)
                target_metric = y.detach().cpu().view(1, -1)
            else:
                resid_metric = resid_flat.view(1, -1)
                target_metric = y.view(1, -1)
            metric = batch_spearmanr(resid_metric, target_metric)
        else:
            resid_metric = resid_flat.view(1, -1)
            target_metric = y.view(1, -1)
            metric = batch_pearsonr(resid_metric, target_metric)
        t4 = time.perf_counter()
        self._ri_func_timing = {
            "ri_func_eval_ms": (t1 - t0) * 1000.0,
            "ri_func_stack_ms": (t2 - t1) * 1000.0,
            "ri_func_lstsq_ms": (t3 - t2) * 1000.0,
            "ri_func_metric_ms": (t4 - t3) * 1000.0,
            "ri_func_total_ms": (t4 - t0) * 1000.0,
            "ri_func_used_k": int(used_k),
            "ri_func_used_sample": int(used_sample),
        }
        return float(metric.mean().item())

    def _calc_ri_func_dispatch(
        self,
        expr: Expression,
        ic_mut: List[float],
        expr_ast_feature: np.ndarray,
    ) -> Tuple[float, Dict[str, Any], Optional[Dict[str, Any]]]:
        del expr_ast_feature
        backend = self.ri_func_backend
        if backend == "corr_cluster":
            score, pending = self._calc_ri_func_corr_cluster(expr)
            return score, {"ri_func_backend": backend}, pending
        if backend == "residual":
            return self._calc_ri_func_v2(expr), {"ri_func_backend": backend}, None
        return self._calc_ri_func(ic_mut), {"ri_func_backend": "mutual_ic"}, None

    def _calc_ri_func_corr_cluster(self, expr: Expression) -> Tuple[float, Dict[str, Any]]:
        t0 = time.perf_counter()
        cluster_idx, similarity, is_new = self._assign_corr_cluster(expr)
        cluster = self._corr_clusters[cluster_idx]
        count = max(1, int(cluster["count"]))
        value_score = max(float(cluster["mean_re"]), 0.0)
        underexplore = 1.0 / (count ** max(self.ri_corr_underexplore_power, 1e-6))
        bonus = self.ri_corr_value_bonus * value_score * underexplore
        if is_new:
            bonus += 0.1 * self.ri_corr_value_bonus
        score = float(bonus - similarity)
        t1 = time.perf_counter()
        self._ri_func_timing = {
            "ri_func_eval_ms": 0.0,
            "ri_func_stack_ms": 0.0,
            "ri_func_lstsq_ms": 0.0,
            "ri_func_metric_ms": (t1 - t0) * 1000.0,
            "ri_func_total_ms": (t1 - t0) * 1000.0,
            "ri_func_used_k": 1,
            "ri_func_used_sample": 0,
        }
        return score, {
            "cluster_idx": int(cluster_idx),
            "cluster_similarity": float(similarity),
            "cluster_is_new": bool(is_new),
        }

    def _assign_corr_cluster(self, expr: Expression) -> Tuple[int, float, bool]:
        if not self._corr_clusters:
            return self._new_corr_cluster(expr), 0.0, True
        best_idx = -1
        best_sim = -1.0
        for idx, cluster in enumerate(self._corr_clusters):
            rep_expr = cluster.get("representative_expr")
            if rep_expr is None:
                continue
            sim = abs(self._get_mutual_ic_cached(expr, rep_expr))
            if np.isnan(sim):
                sim = 0.0
            if sim > best_sim:
                best_idx = idx
                best_sim = sim
        if best_idx < 0 or best_sim < self.ri_corr_threshold:
            return self._new_corr_cluster(expr), max(best_sim, 0.0), True
        return best_idx, best_sim, False

    def _new_corr_cluster(self, expr: Expression) -> int:
        cluster_idx = len(self._corr_clusters)
        self._corr_clusters.append(
            {
                "cluster_id": cluster_idx,
                "representative_expr": expr,
                "representative_expr_key": self._expr_key(expr),
                "count": 0,
                "mean_re": 0.0,
                "positive_re_rate": 0.0,
                "positive_count": 0,
                "last_seen_eval": -1,
                "best_re": float("-inf"),
            }
        )
        return cluster_idx

    def _finalize_corr_cluster(self, pending: Dict[str, Any], expr: Expression, re_value: float) -> Dict[str, Any]:
        cluster_idx = int(pending["cluster_idx"])
        similarity = float(pending.get("cluster_similarity", 0.0))
        is_new = bool(pending.get("cluster_is_new", False))
        cluster = self._corr_clusters[cluster_idx]
        self._update_corr_cluster(cluster_idx, expr, re_value)
        cluster = self._corr_clusters[cluster_idx]
        return {
            "corr_cluster_id": cluster_idx,
            "corr_cluster_count": int(cluster["count"]),
            "corr_cluster_mean_re": float(cluster["mean_re"]),
            "corr_cluster_positive_re_rate": float(cluster["positive_re_rate"]),
            "corr_cluster_similarity": similarity,
            "corr_cluster_is_new": is_new,
        }

    def _update_corr_cluster(self, cluster_idx: int, expr: Expression, re_value: float) -> None:
        cluster = self._corr_clusters[cluster_idx]
        self._update_cluster_stats(cluster, re_value)
        if float(re_value) >= float(cluster.get("best_re", float("-inf"))):
            cluster["best_re"] = float(re_value)
            cluster["representative_expr"] = expr
            cluster["representative_expr_key"] = self._expr_key(expr)

    def _calc_ri_struct_dispatch(
        self,
        expr: Expression,
        token_seq: Optional[List[str]],
        re_value: float,
        expr_ast_feature: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        if self.ri_struct_backend == "ast_cluster":
            return self._calc_ri_struct_ast_v2(expr, expr_ast_feature, re_value)
        if self._use_v2:
            return self._calc_ri_struct_v2(token_seq, re_value)
        return self._calc_ri_struct(token_seq), {
            "cluster_id": -1,
            "cluster_count": 0,
            "cluster_mean_re": 0.0,
            "cluster_positive_re_rate": 0.0,
            "ri_struct_backend": "token_bigram",
        }

    def _calc_ri_struct_ast_v2(
        self,
        expr: Expression,
        expr_ast_feature: np.ndarray,
        re_value: float,
    ) -> Tuple[float, Dict[str, Any]]:
        cluster_idx, similarity, is_new = self._assign_ast_structure_cluster(expr_ast_feature, expr)
        cluster = self._ast_structure_clusters[cluster_idx]
        value_score = max(float(cluster["mean_re"]), 0.0)
        count = max(1, int(cluster["count"]))
        underexplore_score = 1.0 / (count ** max(self.ri_struct_underexplore_power, 1e-6))
        new_cluster_bonus = 0.1 * self.ri_struct_value_bonus if int(cluster["count"]) == 0 else 0.0
        bonus = self.ri_struct_value_bonus * value_score * underexplore_score + new_cluster_bonus
        self._update_ast_structure_cluster(cluster_idx, expr_ast_feature, expr, re_value)
        cluster = self._ast_structure_clusters[cluster_idx]
        return float(bonus), {
            "cluster_id": int(cluster_idx),
            "cluster_count": int(cluster["count"]),
            "cluster_mean_re": float(cluster["mean_re"]),
            "cluster_positive_re_rate": float(cluster["positive_re_rate"]),
            "ast_cluster_id": int(cluster_idx),
            "ast_cluster_count": int(cluster["count"]),
            "ast_cluster_mean_re": float(cluster["mean_re"]),
            "ast_cluster_positive_re_rate": float(cluster["positive_re_rate"]),
            "ast_cluster_similarity": float(similarity),
            "ast_cluster_is_new": bool(is_new),
            "ri_struct_backend": "ast_cluster",
        }

    def _assign_ast_structure_cluster(self, expr_ast_feature: np.ndarray, expr: Expression) -> Tuple[int, float, bool]:
        if not self._ast_structure_clusters:
            return self._new_ast_structure_cluster(expr_ast_feature, expr), 0.0, True
        best_idx = -1
        best_sim = -1.0
        for idx, cluster in enumerate(self._ast_structure_clusters):
            prototype = cluster.get("prototype_feature")
            if prototype is None:
                continue
            sim = cosine_similarity(expr_ast_feature, prototype)
            if sim > best_sim:
                best_idx = idx
                best_sim = sim
        if best_idx < 0 or best_sim < self.ri_ast_similarity_threshold:
            return self._new_ast_structure_cluster(expr_ast_feature, expr), max(best_sim, 0.0), True
        return best_idx, best_sim, False

    def _new_ast_structure_cluster(self, expr_ast_feature: np.ndarray, expr: Expression) -> int:
        cluster_idx = len(self._ast_structure_clusters)
        self._ast_structure_clusters.append(
            {
                "cluster_id": cluster_idx,
                "prototype_feature": np.asarray(expr_ast_feature, dtype=np.float32).copy(),
                "prototype_expr_key": self._expr_key(expr),
                "count": 0,
                "mean_re": 0.0,
                "positive_re_rate": 0.0,
                "positive_count": 0,
                "last_seen_eval": -1,
                "best_re": float("-inf"),
            }
        )
        return cluster_idx

    def _update_ast_structure_cluster(
        self,
        cluster_idx: int,
        expr_ast_feature: np.ndarray,
        expr: Expression,
        re_value: float,
    ) -> None:
        cluster = self._ast_structure_clusters[cluster_idx]
        count = int(cluster["count"])
        prototype = np.asarray(cluster["prototype_feature"], dtype=np.float32)
        expr_ast_feature = np.asarray(expr_ast_feature, dtype=np.float32)
        cluster["prototype_feature"] = ((prototype * count) + expr_ast_feature) / max(1, count + 1)
        self._update_cluster_stats(cluster, re_value)
        if float(re_value) >= float(cluster.get("best_re", float("-inf"))):
            cluster["best_re"] = float(re_value)
            cluster["prototype_expr_key"] = self._expr_key(expr)

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

    def _calc_ri_struct_v2(self, token_seq: Optional[List[str]], re_value: float) -> Tuple[float, Dict[str, Any]]:
        if token_seq is None or len(token_seq) < 2:
            return 0.0, {
                "cluster_id": -1,
                "cluster_count": 0,
                "cluster_mean_re": 0.0,
                "cluster_positive_re_rate": 0.0,
                "ri_struct_backend": "token_bigram",
            }
        new_bigrams = self._token_bigrams(token_seq)
        if not new_bigrams:
            return 0.0, {
                "cluster_id": -1,
                "cluster_count": 0,
                "cluster_mean_re": 0.0,
                "cluster_positive_re_rate": 0.0,
                "ri_struct_backend": "token_bigram",
            }
        cluster_idx = self._assign_structure_cluster(new_bigrams)
        cluster = self._structure_clusters[cluster_idx]
        value_score = max(float(cluster["mean_re"]), 0.0)
        count = max(1, int(cluster["count"]))
        underexplore_score = 1.0 / (count ** max(self.ri_struct_underexplore_power, 1e-6))
        new_cluster_bonus = 0.1 * self.ri_struct_value_bonus if int(cluster["count"]) == 0 else 0.0
        bonus = self.ri_struct_value_bonus * value_score * underexplore_score + new_cluster_bonus
        self._update_structure_cluster(cluster_idx, re_value)
        cluster = self._structure_clusters[cluster_idx]
        return float(bonus), {
            "cluster_id": int(cluster_idx),
            "cluster_count": int(cluster["count"]),
            "cluster_mean_re": float(cluster["mean_re"]),
            "cluster_positive_re_rate": float(cluster["positive_re_rate"]),
            "token_cluster_id": int(cluster_idx),
            "token_cluster_count": int(cluster["count"]),
            "token_cluster_mean_re": float(cluster["mean_re"]),
            "token_cluster_positive_re_rate": float(cluster["positive_re_rate"]),
            "ri_struct_backend": "token_bigram",
        }

    def _calc_ri_reg(self, token_seq: Optional[List[str]]) -> float:
        if token_seq is None:
            return 0.0
        length = len(token_seq)
        if length <= 0:
            return 0.0
        l0 = max(1.0, float(self.ri_reg_l0))
        return -max(0.0, (length - l0) / l0)

    def _calc_ri_reg_v2(self, expr: Expression, token_seq: Optional[List[str]]) -> float:
        if token_seq is None:
            return 0.0
        length = len(token_seq)
        if length <= 0:
            return 0.0
        l0 = max(1.0, float(self.ri_reg_l0))
        length_penalty = max(0.0, (length - l0) / l0)
        depth_penalty = max(0.0, (self._expr_depth(expr) - 4) / 4.0)
        risky_penalty = self._risky_operator_count(expr) / max(1.0, length)
        return -float(length_penalty + depth_penalty + risky_penalty)

    def _calc_pool_turnover_penalty(self) -> Tuple[float, float]:
        make_ensemble = getattr(self.calculator, "make_ensemble_alpha", None)
        if not callable(make_ensemble) or self.size <= 0:
            return math.nan, 0.0
        try:
            with torch.no_grad():
                signal = make_ensemble(self.exprs[:self.size], self.weights[:self.size])
            turnover = self._calc_signal_turnover(signal)
        except Exception:
            return math.nan, 0.0
        if not math.isfinite(turnover):
            return math.nan, 0.0
        penalty = -max(0.0, float(turnover) - self.ri_turnover_baseline)
        return float(turnover), float(penalty)

    def _calc_signal_turnover(self, signal: Tensor) -> float:
        if signal.ndim != 2:
            return math.nan
        n_days, n_stocks = int(signal.shape[0]), int(signal.shape[1])
        if n_days <= 1 or n_stocks <= 0:
            return math.nan
        topk = min(self.ri_turnover_topk, n_stocks)
        prev_hold: Optional[Set[int]] = None
        turnovers: List[float] = []
        stride = max(1, self.ri_turnover_step_stride)
        for day_idx in range(0, n_days, stride):
            day_signal = signal[day_idx]
            finite_mask = torch.isfinite(day_signal)
            valid_idx = torch.nonzero(finite_mask, as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                prev_hold = None
                continue
            day_topk = min(topk, int(valid_idx.numel()))
            valid_scores = day_signal[valid_idx]
            top_positions = torch.topk(valid_scores, k=day_topk, largest=True, sorted=False).indices
            hold_idx = set(valid_idx[top_positions].detach().cpu().tolist())
            if prev_hold is not None and prev_hold:
                overlap = len(prev_hold & hold_idx)
                denom = float(max(1, min(len(prev_hold), len(hold_idx))))
                turnovers.append(1.0 - overlap / denom)
            prev_hold = hold_idx
        if not turnovers:
            return math.nan
        return float(np.mean(turnovers))

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

    def _assign_structure_cluster(self, new_bigrams: Set[Tuple[str, str]]) -> int:
        if not self._structure_clusters:
            self._structure_clusters.append(
                {
                    "cluster_id": 0,
                    "prototype": set(new_bigrams),
                    "count": 0,
                    "mean_re": 0.0,
                    "positive_re_rate": 0.0,
                    "positive_count": 0,
                    "last_seen_eval": -1,
                }
            )
            return 0
        best_idx = -1
        best_sim = -1.0
        for idx, cluster in enumerate(self._structure_clusters):
            old_bigrams = cluster["prototype"]
            inter = len(new_bigrams & old_bigrams)
            union = len(new_bigrams | old_bigrams)
            sim = 0.0 if union == 0 else inter / union
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        threshold = 0.5
        if best_idx < 0 or best_sim < threshold:
            new_idx = len(self._structure_clusters)
            self._structure_clusters.append(
                {
                    "cluster_id": new_idx,
                    "prototype": set(new_bigrams),
                    "count": 0,
                    "mean_re": 0.0,
                    "positive_re_rate": 0.0,
                    "positive_count": 0,
                    "last_seen_eval": -1,
                }
            )
            return new_idx
        return best_idx

    def _update_structure_cluster(self, cluster_idx: int, re_value: float) -> None:
        cluster = self._structure_clusters[cluster_idx]
        self._update_cluster_stats(cluster, re_value)

    def _update_cluster_stats(self, cluster: Dict[str, Any], re_value: float) -> None:
        count = int(cluster["count"])
        positive_count = int(cluster["positive_count"])
        new_count = count + 1
        cluster["mean_re"] = (float(cluster["mean_re"]) * count + float(re_value)) / new_count
        cluster["positive_count"] = positive_count + int(re_value > 0)
        cluster["positive_re_rate"] = float(cluster["positive_count"]) / new_count
        cluster["count"] = new_count
        cluster["last_seen_eval"] = int(self.eval_cnt)

    def _expr_depth(self, expr: Expression) -> int:
        if isinstance(expr, UnaryOperator):
            return 1 + self._expr_depth(expr._operand)
        if isinstance(expr, BinaryOperator):
            return 1 + max(self._expr_depth(expr._lhs), self._expr_depth(expr._rhs))
        if isinstance(expr, RollingOperator):
            return 1 + self._expr_depth(expr._operand)
        if isinstance(expr, PairRollingOperator):
            return 1 + max(self._expr_depth(expr._lhs), self._expr_depth(expr._rhs))
        return 1

    def _risky_operator_count(self, expr: Expression) -> int:
        count = 0
        for node in self._iter_expr_nodes(expr):
            if isinstance(node, (Div, Corr, Cov, Less, Greater)):
                count += 1
            elif node.__class__.__name__ == "Log":
                count += 1
        return count

    @staticmethod
    def _token_bigrams(token_seq: List[str]) -> Set[Tuple[str, str]]:
        return set(zip(token_seq[:-1], token_seq[1:]))
