from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

from alphagen.config import DELTA_TIMES
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
    WMA,
)
from alphagen_qlib.stock_data import FeatureType


HEAD_NAMES = ("base", "trend", "volatility", "volume", "corr", "rank", "explore")
FIELD_NAMES = ("close", "vwap", "open", "high", "low", "volume")
PRICE_FIELDS = ("close", "vwap", "open", "high", "low")
VOLUME_FIELDS = ("volume", "vwap")
WINDOWS = tuple(int(x) for x in DELTA_TIMES)
SMOOTH_OPS = ("mean", "ema", "wma", "std")

FIELD_TYPES = {
    "open": FeatureType.OPEN,
    "close": FeatureType.CLOSE,
    "high": FeatureType.HIGH,
    "low": FeatureType.LOW,
    "volume": FeatureType.VOLUME,
    "vwap": FeatureType.VWAP,
}


def field(name: str) -> Feature:
    return Feature(FIELD_TYPES[name])


def smooth(op: str, expr: Expression, window: int) -> Expression:
    if op == "ema":
        return EMA(expr, window)
    if op == "wma":
        return WMA(expr, window)
    if op == "std":
        return Std(expr, window)
    return Mean(expr, window)


@dataclass(frozen=True)
class MotifSpec:
    motif_id: str
    head: str
    family: str
    build: Callable[[Dict[str, object]], Expression]
    default_field: str = "close"
    default_field2: str = "open"
    default_price: str = "close"
    default_volume: str = "volume"
    default_window: int = 20
    default_smooth_op: str = "mean"
    allow_field: bool = True
    allow_field2: bool = False
    allow_price: bool = False
    allow_volume: bool = False
    allow_window: bool = True
    allow_smooth_op: bool = False
    allow_rank_wrapper: bool = True
    allow_ts_smooth: bool = True
    allow_vol_adjust: bool = True


def _state_field(state: Dict[str, object], key: str, fallback: str) -> Feature:
    return field(str(state.get(key, fallback)))


def _state_window(state: Dict[str, object]) -> int:
    window = int(state.get("window", 20))
    return window if window in WINDOWS else 20


def _apply_wrappers(expr: Expression, state: Dict[str, object]) -> Expression:
    window = _state_window(state)
    if bool(state.get("ts_smooth", False)):
        expr = smooth(str(state.get("smooth_op", "mean")), expr, window)
    if bool(state.get("vol_adjust", False)):
        expr = Div(expr, Std(field("close"), window))
    if bool(state.get("rank_wrapper", False)):
        expr = CSRank(expr)
    return expr


def _build_level(state: Dict[str, object]) -> Expression:
    return _state_field(state, "field", "close")


def _build_mean(state: Dict[str, object]) -> Expression:
    return Mean(_state_field(state, "field", "close"), _state_window(state))


def _build_spread(state: Dict[str, object]) -> Expression:
    return Sub(_state_field(state, "field", "close"), _state_field(state, "field2", "open"))


def _build_intraday_return(state: Dict[str, object]) -> Expression:
    return Div(Sub(field("close"), field("open")), field("open"))


def _build_delta_rank(state: Dict[str, object]) -> Expression:
    return CSRank(Delta(_state_field(state, "field", "close"), _state_window(state)))


def _build_return(state: Dict[str, object]) -> Expression:
    lhs = _state_field(state, "field", "close")
    lag = Ref(_state_field(state, "field", "close"), _state_window(state))
    return Div(Sub(lhs, lag), lag)


def _build_reversal(state: Dict[str, object]) -> Expression:
    lhs = Ref(_state_field(state, "field", "close"), _state_window(state))
    rhs = _state_field(state, "field", "close")
    return CSRank(Sub(lhs, rhs))


def _build_ema_gap(state: Dict[str, object]) -> Expression:
    x = _state_field(state, "field", "close")
    return Sub(x, EMA(_state_field(state, "field", "close"), _state_window(state)))


def _build_std_rank(state: Dict[str, object]) -> Expression:
    return CSRank(Std(_state_field(state, "field", "close"), _state_window(state)))


def _build_range_close(_: Dict[str, object]) -> Expression:
    return Div(Sub(field("high"), field("low")), field("close"))


def _build_delta_std(state: Dict[str, object]) -> Expression:
    return Std(Delta(_state_field(state, "field", "close"), 10), _state_window(state))


def _build_range_volume_mad(state: Dict[str, object]) -> Expression:
    return Mad(Mul(Sub(field("high"), field("low")), field("volume")), _state_window(state))


def _build_volume_ratio(state: Dict[str, object]) -> Expression:
    x = _state_field(state, "volume_field", "volume")
    return Div(x, Mean(_state_field(state, "volume_field", "volume"), _state_window(state)))


def _build_vwap_gap(_: Dict[str, object]) -> Expression:
    return Sub(field("vwap"), field("close"))


def _build_volume_momentum(state: Dict[str, object]) -> Expression:
    x = _state_field(state, "volume_field", "volume")
    return Div(Delta(x, _state_window(state)), Mean(_state_field(state, "volume_field", "volume"), _state_window(state)))


def _build_price_volume_corr(state: Dict[str, object]) -> Expression:
    price = _state_field(state, "price_field", "close")
    volume = _state_field(state, "volume_field", "volume")
    return Corr(CSRank(price), CSRank(volume), _state_window(state))


def _build_cov_fields(state: Dict[str, object]) -> Expression:
    return Cov(_state_field(state, "field", "close"), _state_field(state, "field2", "volume"), _state_window(state))


def _build_range_volume_corr(state: Dict[str, object]) -> Expression:
    return Corr(Sub(field("high"), field("low")), field("volume"), _state_window(state))


def _build_rank_field(state: Dict[str, object]) -> Expression:
    return CSRank(_state_field(state, "field", "close"))


def _build_tsrank_field(state: Dict[str, object]) -> Expression:
    return TSRank(_state_field(state, "field", "close"), _state_window(state))


def _build_relative_mean(state: Dict[str, object]) -> Expression:
    x = _state_field(state, "field", "close")
    return Greater(x, Mean(_state_field(state, "field", "close"), _state_window(state)))


def _build_smooth_rank(state: Dict[str, object]) -> Expression:
    return CSRank(smooth(str(state.get("smooth_op", "wma")), _state_field(state, "field", "vwap"), _state_window(state)))


def _build_vol_adjusted_delta(state: Dict[str, object]) -> Expression:
    return Div(Delta(_state_field(state, "field", "vwap"), _state_window(state)), Std(field("close"), _state_window(state)))


MOTIF_BANK: Tuple[MotifSpec, ...] = (
    MotifSpec("base_level", "base", "simple", _build_level, allow_rank_wrapper=True, allow_ts_smooth=True, allow_vol_adjust=False),
    MotifSpec("base_mean", "base", "simple", _build_mean, allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=True),
    MotifSpec("base_spread", "base", "relative", _build_spread, default_field="close", default_field2="open", allow_field2=True, allow_ts_smooth=True),
    MotifSpec("base_intraday_return", "base", "return", _build_intraday_return, allow_field=False, allow_field2=False, allow_window=False, allow_rank_wrapper=True, allow_ts_smooth=True),
    MotifSpec("trend_delta_rank", "trend", "trend", _build_delta_rank, default_field="close", allow_rank_wrapper=False, allow_ts_smooth=False),
    MotifSpec("trend_return", "trend", "trend", _build_return, default_field="close", allow_rank_wrapper=True, allow_ts_smooth=True),
    MotifSpec("trend_reversal_rank", "trend", "trend", _build_reversal, default_field="close", allow_rank_wrapper=False, allow_ts_smooth=False),
    MotifSpec("trend_ema_gap", "trend", "trend", _build_ema_gap, default_field="vwap", allow_smooth_op=False, allow_rank_wrapper=True),
    MotifSpec("vol_std_rank", "volatility", "volatility", _build_std_rank, default_field="close", allow_rank_wrapper=False, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("vol_range_close", "volatility", "volatility", _build_range_close, allow_field=False, allow_field2=False, allow_window=False, allow_rank_wrapper=True, allow_ts_smooth=True, allow_vol_adjust=False),
    MotifSpec("vol_delta_std", "volatility", "volatility", _build_delta_std, default_field="close", allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("vol_range_volume_mad", "volatility", "volume", _build_range_volume_mad, allow_field=False, allow_field2=False, allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("volume_ratio", "volume", "volume", _build_volume_ratio, allow_field=False, allow_field2=False, allow_volume=True, allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("volume_vwap_gap", "volume", "volume", _build_vwap_gap, allow_field=False, allow_field2=False, allow_window=False, allow_rank_wrapper=True, allow_ts_smooth=True),
    MotifSpec("volume_momentum", "volume", "volume", _build_volume_momentum, allow_field=False, allow_field2=False, allow_volume=True, allow_rank_wrapper=True, allow_ts_smooth=False),
    MotifSpec("corr_price_volume_rank", "corr", "corr", _build_price_volume_corr, allow_field=False, allow_field2=False, allow_price=True, allow_volume=True, allow_rank_wrapper=False, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("corr_cov_fields", "corr", "corr", _build_cov_fields, default_field="close", default_field2="volume", allow_field2=True, allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("corr_range_volume", "corr", "corr", _build_range_volume_corr, allow_field=False, allow_field2=False, allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("rank_field", "rank", "rank", _build_rank_field, allow_rank_wrapper=False, allow_ts_smooth=False, allow_vol_adjust=True),
    MotifSpec("rank_tsrank", "rank", "rank", _build_tsrank_field, allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("rank_relative_mean", "rank", "rank", _build_relative_mean, allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("explore_smooth_rank", "explore", "explore", _build_smooth_rank, default_field="vwap", default_smooth_op="wma", allow_smooth_op=True, allow_rank_wrapper=False, allow_ts_smooth=False),
    MotifSpec("explore_vol_adjusted_delta", "explore", "explore", _build_vol_adjusted_delta, default_field="vwap", allow_rank_wrapper=True, allow_ts_smooth=False, allow_vol_adjust=False),
    MotifSpec("explore_price_volume_corr", "explore", "corr", _build_price_volume_corr, default_price="vwap", allow_price=True, allow_volume=True, allow_rank_wrapper=False, allow_ts_smooth=False, allow_vol_adjust=False),
)

MOTIFS_BY_HEAD: Dict[str, List[int]] = {head: [] for head in HEAD_NAMES}
for idx, motif in enumerate(MOTIF_BANK):
    MOTIFS_BY_HEAD.setdefault(motif.head, []).append(idx)


def default_state(motif_index: int) -> Dict[str, object]:
    motif = MOTIF_BANK[motif_index]
    return {
        "motif_index": int(motif_index),
        "field": motif.default_field,
        "field2": motif.default_field2,
        "price_field": motif.default_price,
        "volume_field": motif.default_volume,
        "window": int(motif.default_window),
        "smooth_op": motif.default_smooth_op,
        "rank_wrapper": False,
        "ts_smooth": False,
        "vol_adjust": False,
    }


def build_motif_expression(state: Dict[str, object]) -> Expression:
    motif_index = int(state["motif_index"])
    motif = MOTIF_BANK[motif_index]
    return _apply_wrappers(motif.build(state), state)


def motif_summary() -> Dict[str, object]:
    return {
        "motif_count": len(MOTIF_BANK),
        "head_counts": dict(Counter(motif.head for motif in MOTIF_BANK)),
        "family_counts": dict(Counter(motif.family for motif in MOTIF_BANK)),
    }


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


def expression_stats(expr: Expression) -> Dict[str, object]:
    counts: Counter[str] = Counter()
    constant_count = 0
    max_depth = 0
    repeated_unary = 0
    comparison_inside_arithmetic = 0

    def walk(node: Expression, depth: int, parent: str = "") -> None:
        nonlocal constant_count, max_depth, repeated_unary, comparison_inside_arithmetic
        max_depth = max(max_depth, depth)
        name = type(node).__name__
        if isinstance(node, Constant):
            constant_count += 1
        if isinstance(node, (UnaryOperator, RollingOperator, PairRollingOperator, BinaryOperator)):
            counts[name] += 1
        if parent == name and name in {"CSRank", "TSRank", "Log", "SafeSqrt"}:
            repeated_unary += 1
        if name in {"Greater", "Less"} and parent in {"Add", "Sub", "Mul", "Div"}:
            comparison_inside_arithmetic += 1
        for child in _children(node):
            walk(child, depth + 1, name)

    walk(expr, 1)
    op_total = sum(counts.values())
    return {
        "op_counts": dict(counts),
        "constant_count": constant_count,
        "depth": max_depth,
        "repeated_unary": repeated_unary,
        "comparison_inside_arithmetic": comparison_inside_arithmetic,
        "op_total": op_total,
    }


def naturalness_score(expr: Expression) -> Dict[str, object]:
    stats = expression_stats(expr)
    counts = Counter(stats["op_counts"])
    reasons: List[str] = []
    penalty = 0.0

    div_count = int(counts.get("Div", 0))
    if div_count > 2:
        reasons.append("too_many_div")
        penalty += 0.25 * (div_count - 2)
    if int(stats["constant_count"]) > 1:
        reasons.append("too_many_constants")
        penalty += 0.25 * (int(stats["constant_count"]) - 1)
    if int(stats["depth"]) > 8:
        reasons.append("too_deep")
        penalty += 0.15 * (int(stats["depth"]) - 8)
    if int(stats["repeated_unary"]) > 0:
        reasons.append("repeated_unary_wrapper")
        penalty += 0.5
    if int(stats["comparison_inside_arithmetic"]) > 0:
        reasons.append("comparison_inside_arithmetic")
        penalty += 0.5
    if int(counts.get("Log", 0)) > 0:
        reasons.append("log_present")
        penalty += 0.2
    if int(counts.get("SafeSqrt", 0)) > 1:
        reasons.append("repeated_safesqrt")
        penalty += 0.25

    score = max(0.0, 1.0 - penalty)
    hard_fail = (
        div_count > 3
        or int(stats["constant_count"]) > 2
        or int(stats["depth"]) > 10
        or int(stats["repeated_unary"]) > 0
        or int(stats["comparison_inside_arithmetic"]) > 0
    )
    return {
        "score": float(score),
        "passed": bool(not hard_fail and score >= 0.35),
        "reasons": reasons,
        "stats": stats,
    }
