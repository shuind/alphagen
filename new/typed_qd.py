from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import (
    Abs,
    Add,
    BinaryOperator,
    CSRank,
    Constant,
    Corr,
    Cov,
    Delta,
    Div,
    EMA,
    Expression,
    Feature,
    Greater,
    Less,
    Log,
    Mad,
    Max,
    Mean,
    Min,
    Mul,
    PairRollingOperator,
    Ref,
    RollingOperator,
    SafeSqrt,
    Sign,
    Std,
    Sub,
    TSRank,
    UnaryOperator,
    Var,
    WMA,
)
from alphagen.models.alpha_pool import AlphaPool
from alphagen_qlib.stock_data import FeatureType


PRICE_FEATURES = {FeatureType.OPEN, FeatureType.CLOSE, FeatureType.HIGH, FeatureType.LOW, FeatureType.VWAP}
VOLUME_FEATURES = {FeatureType.VOLUME}
ARITHMETIC_OPS = {"Add", "Sub", "Mul", "Div"}
RISKY_OPS = {"Div", "Log", "SafeSqrt", "Corr", "Cov", "Std", "Var", "Mad"}


def _children(expr: Expression) -> Iterable[Expression]:
    if isinstance(expr, UnaryOperator):
        yield expr._operand
    elif isinstance(expr, RollingOperator):
        yield expr._operand
    elif isinstance(expr, PairRollingOperator):
        yield expr._lhs
        yield expr._rhs
    elif isinstance(expr, BinaryOperator):
        yield expr._lhs
        yield expr._rhs


def _merge_types(lhs: str, rhs: str, op_name: str, reasons: List[str]) -> str:
    pair = {lhs, rhs}
    if op_name in {"Add", "Sub"} and "price" in pair and "volume" in pair:
        reasons.append("price_volume_add_sub")
    if "boolean" in pair and op_name in ARITHMETIC_OPS:
        reasons.append("boolean_inside_arithmetic")
    if op_name == "Div":
        if rhs in {"constant", "boolean"}:
            reasons.append("bad_div_denominator")
        if lhs == "volume" and rhs == "price":
            return "volume"
        if lhs == "price" and rhs == "price":
            return "return"
        if lhs == "volume" and rhs == "volume":
            return "volume"
        return "numeric"
    if op_name == "Mul" and "volume" in pair and ("price" in pair or "return" in pair):
        return "volume"
    if lhs == rhs:
        return lhs
    if "rank" in pair:
        return "rank"
    if "return" in pair:
        return "return"
    if "volatility" in pair:
        return "volatility"
    if "price" in pair:
        return "price"
    if "volume" in pair:
        return "volume"
    return "numeric"


def typed_dsl_check(expr: Expression) -> Dict[str, Any]:
    op_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    constant_count = 0
    max_depth = 0
    repeated_wrappers = 0
    reasons: List[str] = []

    def walk(node: Expression, depth: int, parent: str = "") -> str:
        nonlocal constant_count, max_depth, repeated_wrappers
        max_depth = max(max_depth, depth)
        name = type(node).__name__
        if isinstance(node, Feature):
            feature = node._feature
            field_counts[f"${feature.name.lower()}"] += 1
            if feature in PRICE_FEATURES:
                return "price"
            if feature in VOLUME_FEATURES:
                return "volume"
            return "numeric"
        if isinstance(node, Constant):
            constant_count += 1
            return "constant"
        if isinstance(node, (UnaryOperator, RollingOperator, PairRollingOperator, BinaryOperator)):
            op_counts[name] += 1
        if parent == name and name in {"CSRank", "TSRank", "Log", "SafeSqrt", "Sign"}:
            repeated_wrappers += 1
            reasons.append(f"repeated_{name}")

        if isinstance(node, CSRank):
            walk(node._operand, depth + 1, name)
            return "rank"
        if isinstance(node, Sign):
            walk(node._operand, depth + 1, name)
            return "boolean"
        if isinstance(node, (Abs, SafeSqrt, Log)):
            child_type = walk(node._operand, depth + 1, name)
            return "numeric" if child_type in {"constant", "boolean"} else child_type
        if isinstance(node, (Ref, Mean, EMA, WMA)):
            return walk(node._operand, depth + 1, name)
        if isinstance(node, Delta):
            child_type = walk(node._operand, depth + 1, name)
            return "return" if child_type == "price" else child_type
        if isinstance(node, (Std, Var, Mad)):
            walk(node._operand, depth + 1, name)
            return "volatility"
        if isinstance(node, (Max, Min)):
            return walk(node._operand, depth + 1, name)
        if isinstance(node, TSRank):
            walk(node._operand, depth + 1, name)
            return "rank"
        if isinstance(node, (Corr, Cov)):
            lhs_type = walk(node._lhs, depth + 1, name)
            rhs_type = walk(node._rhs, depth + 1, name)
            if "boolean" in {lhs_type, rhs_type}:
                reasons.append("boolean_inside_corr_cov")
            return "correlation"
        if isinstance(node, (Greater, Less)):
            walk(node._lhs, depth + 1, name)
            walk(node._rhs, depth + 1, name)
            return "boolean"
        if isinstance(node, BinaryOperator):
            lhs_type = walk(node._lhs, depth + 1, name)
            rhs_type = walk(node._rhs, depth + 1, name)
            return _merge_types(lhs_type, rhs_type, name, reasons)
        return "numeric"

    root_type = walk(expr, 1)
    div_count = op_counts.get("Div", 0)
    log_count = op_counts.get("Log", 0)
    safesqrt_count = op_counts.get("SafeSqrt", 0)

    if root_type in {"boolean", "constant"}:
        reasons.append(f"bad_root_type:{root_type}")
    if constant_count > 1:
        reasons.append("constant_abuse")
    if div_count > 2:
        reasons.append("too_many_div")
    if log_count > 1:
        reasons.append("too_many_log")
    if safesqrt_count > 1:
        reasons.append("too_many_safesqrt")
    if max_depth > 8:
        reasons.append("too_deep")
    if repeated_wrappers > 0:
        reasons.append("repeated_wrapper")

    valid = len(reasons) == 0
    stats = {
        "root_type": root_type,
        "op_counts": dict(op_counts),
        "field_counts": dict(field_counts),
        "constant_count": int(constant_count),
        "depth": int(max_depth),
        "div_count": int(div_count),
        "log_count": int(log_count),
        "safesqrt_count": int(safesqrt_count),
        "node_count": int(sum(op_counts.values()) + sum(field_counts.values()) + constant_count),
        "risky_op_count": int(sum(op_counts.get(op, 0) for op in RISKY_OPS)),
    }
    return {"valid": bool(valid), "reasons": reasons, "stats": stats}


def expression_style(type_stats: Dict[str, Any]) -> str:
    op_counts = Counter(type_stats.get("op_counts", {}))
    fields = Counter(type_stats.get("field_counts", {}))
    if op_counts.get("Corr", 0) or op_counts.get("Cov", 0):
        return "corr"
    if "$volume" in fields:
        return "volume"
    if op_counts.get("Std", 0) or op_counts.get("Var", 0) or op_counts.get("Mad", 0):
        return "volatility"
    if op_counts.get("CSRank", 0) or op_counts.get("TSRank", 0):
        return "rank"
    if op_counts.get("Delta", 0) or op_counts.get("Ref", 0) or op_counts.get("EMA", 0) or op_counts.get("WMA", 0):
        return "trend"
    return str(type_stats.get("root_type", "base"))


def complexity_bin(type_stats: Dict[str, Any]) -> str:
    nodes = int(type_stats.get("node_count", 0))
    depth = int(type_stats.get("depth", 0))
    if nodes <= 6 and depth <= 4:
        return "low"
    if nodes <= 11 and depth <= 6:
        return "mid"
    return "high"


def robust_rankic_score(
    expr: Expression,
    calculators: List[AlphaCalculator],
    robust_lambda: float = 0.5,
    bottom_k: int = 0,
) -> Tuple[float, List[float], Dict[str, float]]:
    values: List[float] = []
    for calc in calculators:
        try:
            value = float(calc.calc_single_rIC_ret(expr))
        except Exception:
            value = math.nan
        if math.isfinite(value):
            values.append(value)
    if not values:
        return -1.0, [], {"mean": math.nan, "std": math.nan, "bottom_mean": math.nan}
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if bottom_k > 0:
        k = min(int(bottom_k), len(values))
        score = float(np.sort(arr)[:k].mean())
        bottom_mean = score
    else:
        score = float(mean - float(robust_lambda) * std)
        bottom_mean = float(np.sort(arr)[: max(1, min(2, len(values)))].mean())
    return score, values, {"mean": mean, "std": std, "bottom_mean": bottom_mean}


@dataclass
class QDEntry:
    score: float
    expr: Expression
    descriptor: str
    metadata: Dict[str, Any]


class QualityDiversityArchive:
    def __init__(self, cell_capacity: int = 2, behavior_threshold: float = 0.7) -> None:
        self.cell_capacity = max(1, int(cell_capacity))
        self.behavior_threshold = float(behavior_threshold)
        self.cells: Dict[str, List[QDEntry]] = {}
        self.behavior_prototypes: List[Expression] = []

    def assign_behavior_cluster(self, expr: Expression, mutual_fn: Callable[[Expression, Expression], float]) -> Tuple[int, bool, float]:
        best_idx = -1
        best_sim = -1.0
        for idx, proto in enumerate(self.behavior_prototypes):
            try:
                sim = abs(float(mutual_fn(expr, proto)))
            except Exception:
                sim = 0.0
            if sim > best_sim:
                best_idx = idx
                best_sim = sim
        if best_idx >= 0 and best_sim >= self.behavior_threshold:
            return best_idx, False, float(best_sim)
        return len(self.behavior_prototypes), True, float(max(0.0, best_sim))

    def descriptor(self, style: str, complexity: str, behavior_cluster: int) -> str:
        return f"{style}|{complexity}|b{behavior_cluster}"

    def would_accept(self, descriptor: str, score: float) -> Tuple[bool, int]:
        entries = self.cells.get(descriptor, [])
        if len(entries) < self.cell_capacity:
            return True, -1
        worst_idx = min(range(len(entries)), key=lambda idx: entries[idx].score)
        return float(score) > entries[worst_idx].score, worst_idx

    def commit(
        self,
        expr: Expression,
        descriptor: str,
        score: float,
        metadata: Dict[str, Any],
        behavior_is_new: bool,
        replace_idx: int = -1,
    ) -> None:
        if behavior_is_new:
            self.behavior_prototypes.append(expr)
        entries = self.cells.setdefault(descriptor, [])
        entry = QDEntry(score=float(score), expr=expr, descriptor=descriptor, metadata=dict(metadata))
        if len(entries) < self.cell_capacity:
            entries.append(entry)
            entries.sort(key=lambda item: item.score, reverse=True)
            return
        if replace_idx >= 0:
            entries[replace_idx] = entry
            entries.sort(key=lambda item: item.score, reverse=True)

    def summary(self) -> Dict[str, Any]:
        cell_sizes = {cell: len(entries) for cell, entries in self.cells.items()}
        return {
            "coverage": len(self.cells),
            "total_entries": int(sum(cell_sizes.values())),
            "behavior_cluster_count": len(self.behavior_prototypes),
            "cell_capacity": self.cell_capacity,
            "behavior_threshold": self.behavior_threshold,
            "cell_sizes": cell_sizes,
            "top_cells": [
                (cell, entries[0].score if entries else math.nan)
                for cell, entries in sorted(self.cells.items(), key=lambda item: item[1][0].score if item[1] else -999, reverse=True)[:12]
            ],
        }


class TypedQDAlphaPool(AlphaPool):
    def __init__(
        self,
        *args,
        robust_calculators: Optional[List[AlphaCalculator]] = None,
        robust_lambda: float = 0.5,
        robust_bottom_k: int = 0,
        qd_cell_capacity: int = 2,
        qd_behavior_threshold: float = 0.7,
        qd_bonus: float = 0.02,
        min_robust_score: float = -1.0,
        use_robust_reward: bool = True,
        use_qd_archive: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, reward_mode="re", lambda_ri=0.0, **kwargs)
        self.robust_calculators = list(robust_calculators or [self.calculator])
        self.robust_lambda = float(robust_lambda)
        self.robust_bottom_k = int(robust_bottom_k)
        self.qd_bonus = float(qd_bonus)
        self.min_robust_score = float(min_robust_score)
        self.use_robust_reward = bool(use_robust_reward)
        self.use_qd_archive = bool(use_qd_archive)
        self.archive = QualityDiversityArchive(qd_cell_capacity, qd_behavior_threshold)

        self.expr_qd_descriptors: List[Optional[str]] = [None for _ in range(self.capacity + 1)]
        self.expr_robust_scores: List[Optional[float]] = [None for _ in range(self.capacity + 1)]
        self.expr_yearly_rankics: List[Optional[List[float]]] = [None for _ in range(self.capacity + 1)]
        self.expr_type_roots: List[Optional[str]] = [None for _ in range(self.capacity + 1)]
        self.expr_type_styles: List[Optional[str]] = [None for _ in range(self.capacity + 1)]
        self.expr_behavior_clusters: List[Optional[int]] = [None for _ in range(self.capacity + 1)]

        self.type_reject_counts = Counter()
        self.qd_reject_counts = Counter()
        self._pending_descriptor = ""
        self._pending_robust_score = 0.0
        self._pending_yearly_rankics: List[float] = []
        self._pending_root_type = ""
        self._pending_style = ""
        self._pending_behavior_cluster = -1

    def try_new_expr(self, expr: Expression, token_seq: Optional[List[str]] = None, source_head: str = "base") -> Tuple[float, Dict]:
        typed = typed_dsl_check(expr)
        if not typed["valid"]:
            for reason in typed["reasons"]:
                self.type_reject_counts[reason] += 1
            info = {
                "invalid": True,
                "invalid_reason": "typed_dsl_failed",
                "typed_valid": False,
                "typed_reasons": typed["reasons"],
                "typed_stats": typed["stats"],
                "source_head": source_head,
                "reward_total": -1.0,
                "reward_pool": 0.0,
                "re": 0.0,
            }
            self.last_reward_info = info
            return -1.0, info

        if self.use_robust_reward or self.use_qd_archive:
            robust_score, yearly_rankics, robust_stats = robust_rankic_score(
                expr,
                calculators=self.robust_calculators,
                robust_lambda=self.robust_lambda,
                bottom_k=self.robust_bottom_k,
            )
        else:
            robust_score, yearly_rankics, robust_stats = 0.0, [], {"mean": math.nan, "std": math.nan, "bottom_mean": math.nan}
        style = expression_style(typed["stats"])
        comp_bin = complexity_bin(typed["stats"])
        if self.use_qd_archive:
            behavior_cluster, behavior_is_new, behavior_similarity = self.archive.assign_behavior_cluster(
                expr,
                self._get_mutual_ic_cached,
            )
        else:
            behavior_cluster, behavior_is_new, behavior_similarity = -1, False, 0.0
        descriptor = self.archive.descriptor(style, comp_bin, behavior_cluster)

        accepted_by_score = (not (self.use_robust_reward or self.use_qd_archive)) or robust_score >= self.min_robust_score
        qd_accept, replace_idx = self.archive.would_accept(descriptor, robust_score) if self.use_qd_archive else (True, -1)
        if not accepted_by_score or not qd_accept:
            reason = "below_min_robust" if not accepted_by_score else "cell_not_improved"
            self.qd_reject_counts[reason] += 1
            info = {
                "invalid": False,
                "typed_valid": True,
                "typed_stats": typed["stats"],
                "typed_style": style,
                "typed_complexity_bin": comp_bin,
                "robust_score": float(robust_score),
                "yearly_rankics": yearly_rankics,
                "robust_stats": robust_stats,
                "qd_descriptor": descriptor,
                "qd_accepted": False,
                "qd_reject_reason": reason,
                "behavior_cluster": int(behavior_cluster),
                "behavior_cluster_is_new": bool(behavior_is_new),
                "behavior_similarity": float(behavior_similarity),
                "source_head": source_head,
                "reward_total": float(robust_score - 0.05),
                "reward_pool": 0.0,
                "re": 0.0,
            }
            self.last_reward_info = info
            return float(robust_score - 0.05), info

        self._pending_descriptor = descriptor
        self._pending_robust_score = float(robust_score)
        self._pending_yearly_rankics = list(yearly_rankics)
        self._pending_root_type = str(typed["stats"].get("root_type", ""))
        self._pending_style = style
        self._pending_behavior_cluster = int(behavior_cluster)

        reward, info = super().try_new_expr(expr, token_seq=token_seq)
        info = dict(info)
        if info.get("invalid", False):
            info.update(
                {
                    "typed_valid": True,
                    "typed_stats": typed["stats"],
                    "robust_score": float(robust_score),
                    "yearly_rankics": yearly_rankics,
                    "qd_descriptor": descriptor,
                    "qd_accepted": False,
                    "qd_reject_reason": "pool_invalid",
                    "source_head": source_head,
                }
            )
            self.last_reward_info = info
            return reward, info

        qd_reward = self.qd_bonus * (1.0 if behavior_is_new else 0.25) if self.use_qd_archive else 0.0
        robust_reward = robust_score if self.use_robust_reward else 0.0
        reward_total = float(reward + robust_reward + qd_reward)
        archive_meta = {
            "source_head": source_head,
            "style": style,
            "complexity_bin": comp_bin,
            "yearly_rankics": yearly_rankics,
            "root_type": typed["stats"].get("root_type", ""),
        }
        if self.use_qd_archive:
            self.archive.commit(
                expr=expr,
                descriptor=descriptor,
                score=robust_score,
                metadata=archive_meta,
                behavior_is_new=behavior_is_new,
                replace_idx=replace_idx,
            )
        info.update(
            {
                "source_head": source_head,
                "typed_valid": True,
                "typed_stats": typed["stats"],
                "typed_style": style,
                "typed_complexity_bin": comp_bin,
                "robust_score": float(robust_score),
                "yearly_rankics": yearly_rankics,
                "robust_stats": robust_stats,
                "qd_descriptor": descriptor,
                "qd_accepted": True,
                "qd_reward": float(qd_reward),
                "robust_reward": float(robust_reward),
                "qd_bonus": float(self.qd_bonus),
                "typed_use_robust_reward": bool(self.use_robust_reward),
                "typed_use_qd_archive": bool(self.use_qd_archive),
                "behavior_cluster": int(behavior_cluster),
                "behavior_cluster_is_new": bool(behavior_is_new),
                "behavior_similarity": float(behavior_similarity),
                "reward_total": reward_total,
                "reward_pool": float(info.get("reward_pool", reward)),
                "qd_archive": self.archive.summary(),
                "typed_reject_counts": dict(self.type_reject_counts),
                "qd_reject_counts": dict(self.qd_reject_counts),
            }
        )
        self.last_reward_info = info
        return reward_total, info

    def _add_factor(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super()._add_factor(*args, **kwargs)
        idx = self.size - 1
        self.expr_qd_descriptors[idx] = self._pending_descriptor
        self.expr_robust_scores[idx] = float(self._pending_robust_score)
        self.expr_yearly_rankics[idx] = list(self._pending_yearly_rankics)
        self.expr_type_roots[idx] = self._pending_root_type
        self.expr_type_styles[idx] = self._pending_style
        self.expr_behavior_clusters[idx] = int(self._pending_behavior_cluster)

    def _swap_idx(self, i, j) -> None:
        super()._swap_idx(i, j)
        self.expr_qd_descriptors[i], self.expr_qd_descriptors[j] = self.expr_qd_descriptors[j], self.expr_qd_descriptors[i]
        self.expr_robust_scores[i], self.expr_robust_scores[j] = self.expr_robust_scores[j], self.expr_robust_scores[i]
        self.expr_yearly_rankics[i], self.expr_yearly_rankics[j] = self.expr_yearly_rankics[j], self.expr_yearly_rankics[i]
        self.expr_type_roots[i], self.expr_type_roots[j] = self.expr_type_roots[j], self.expr_type_roots[i]
        self.expr_type_styles[i], self.expr_type_styles[j] = self.expr_type_styles[j], self.expr_type_styles[i]
        self.expr_behavior_clusters[i], self.expr_behavior_clusters[j] = self.expr_behavior_clusters[j], self.expr_behavior_clusters[i]

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload["qd_descriptors"] = list(self.expr_qd_descriptors[: self.size])
        payload["robust_scores"] = list(self.expr_robust_scores[: self.size])
        payload["yearly_rankics"] = list(self.expr_yearly_rankics[: self.size])
        payload["type_roots"] = list(self.expr_type_roots[: self.size])
        payload["type_styles"] = list(self.expr_type_styles[: self.size])
        payload["behavior_clusters"] = list(self.expr_behavior_clusters[: self.size])
        payload["typed_reject_counts"] = dict(self.type_reject_counts)
        payload["qd_reject_counts"] = dict(self.qd_reject_counts)
        payload["qd_archive"] = self.archive.summary()
        payload["typed_qd_config"] = {
            "robust_lambda": self.robust_lambda,
            "robust_bottom_k": self.robust_bottom_k,
            "qd_bonus": self.qd_bonus,
            "min_robust_score": self.min_robust_score,
            "use_robust_reward": self.use_robust_reward,
            "use_qd_archive": self.use_qd_archive,
        }
        return payload
