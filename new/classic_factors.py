import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from alphagen.config import CONSTANTS, DELTA_TIMES, MAX_EXPR_LENGTH, OPERATORS
from alphagen.data.expression import PairRollingOperator, RollingOperator
from alphagen.data.tokens import ConstantToken, DeltaTimeToken, FeatureToken, OperatorToken, Token
from alphagen.data.tree import ExpressionBuilder
from alphagen.rl.env.wrapper import OFFSET_CONSTANT, OFFSET_DELTA_TIME, OFFSET_FEATURE, OFFSET_OP
from alphagen_qlib.stock_data import FeatureType
from new.env import HEAD_NAMES


NUMBER_PATTERN = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?$", re.IGNORECASE)
CONSTANT_PATTERN = re.compile(r"^Constant\(([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\)$", re.IGNORECASE)

OP_BY_NAME = {op.__name__: op for op in OPERATORS}
HEAD_SET = set(HEAD_NAMES)
DEFAULT_LOSS_WEIGHTS = "next=1.0,head=0.2,attr=0.2"
HEAD_ALIASES = {
    "simple": "base",
    "ts": "trend",
    "pv": "volume",
}


def _infer_head(record: Dict[str, str]) -> str:
    head = HEAD_ALIASES.get((record.get("head") or "").strip(), (record.get("head") or "").strip())
    tokens = record.get("tokens", "")
    family = (record.get("family") or "").lower()
    tags = (record.get("tags") or "").lower()
    if head == "explore":
        return head
    if "CSRank" in tokens or "TSRank" in tokens or " Sign" in f" {tokens} " or family == "relative" or "rank" in tags:
        return "rank"
    if " Corr" in f" {tokens} " or " Cov" in f" {tokens} " or "corr" in tags or "cov" in tags:
        return "corr"
    if "$volume" in tokens or family == "pv" or "volume" in tags or "moneyflow" in tags:
        return "volume"
    if (
        "Std" in tokens
        or "Var" in tokens
        or "Mad" in tokens
        or "SafeSqrt" in tokens
        or "volatility" in tags
        or "range" in tags
        or "kline" in tags
    ):
        return "volatility"
    if (
        "Delta" in tokens
        or "Ref" in tokens
        or "Mean" in tokens
        or "EMA" in tokens
        or "WMA" in tokens
        or "momentum" in tags
        or "return" in tags
        or "reversal" in tags
        or family == "ts"
    ):
        return "trend"
    return head if head in HEAD_SET else "base"


@dataclass(frozen=True)
class ClassicFactor:
    name: str
    head: str
    token_text: List[str]
    action_ids: List[int]
    tags: Tuple[str, ...] = ()
    family: str = ""
    weight: float = 1.0
    source: str = ""
    comment: str = ""
    augmented_from: str = ""


@dataclass(frozen=True)
class SkippedFactor:
    name: str
    head: str
    reason: str
    family: str = ""
    source: str = ""


@dataclass(frozen=True)
class ClassicFactorLoadResult:
    factors: List[ClassicFactor]
    skipped: List[SkippedFactor]
    stats: Dict


BUILTIN_V1_CLASSIC_FACTORS: List[Dict[str, str]] = [
    {"name": "close_open_return", "head": "base", "tokens": "$close $open Sub $open Div", "tags": "return,simple", "family": "simple", "source": "builtin_v1"},
    {"name": "high_low_spread", "head": "base", "tokens": "$high $low Sub $close Div", "tags": "range", "family": "simple", "source": "builtin_v1"},
    {"name": "vwap_close_gap", "head": "base", "tokens": "$vwap $close Sub $close Div", "tags": "price", "family": "simple", "source": "builtin_v1"},
    {"name": "price_momentum_20", "head": "base", "tokens": "$close d:20 Delta $close Div", "tags": "momentum", "family": "ts", "source": "builtin_v1"},
    {"name": "intraday_position", "head": "base", "tokens": "$close $low Sub $high $low Sub Div", "tags": "range", "family": "simple", "source": "builtin_v1"},
    {"name": "volume_momentum_20", "head": "base", "tokens": "$volume d:20 Delta $volume Div", "tags": "volume", "family": "pv", "source": "builtin_v1"},
    {"name": "close_minus_open", "head": "simple", "tokens": "$close $open Sub", "tags": "simple", "family": "simple", "source": "builtin_v1"},
    {"name": "high_minus_low", "head": "simple", "tokens": "$high $low Sub", "tags": "simple", "family": "simple", "source": "builtin_v1"},
    {"name": "close_div_vwap", "head": "simple", "tokens": "$close $vwap Div", "tags": "simple", "family": "simple", "source": "builtin_v1"},
    {"name": "abs_intraday_return", "head": "simple", "tokens": "$close $open Sub Abs", "tags": "simple", "family": "simple", "source": "builtin_v1"},
    {"name": "log_volume", "head": "simple", "tokens": "$volume Log", "tags": "simple,volume", "family": "simple", "source": "builtin_v1"},
    {"name": "close_low_gap", "head": "simple", "tokens": "$close $low Sub", "tags": "simple", "family": "simple", "source": "builtin_v1"},
    {"name": "mean_close_20", "head": "ts", "tokens": "$close d:20 Mean", "tags": "ts,mean", "family": "ts", "source": "builtin_v1"},
    {"name": "std_close_20", "head": "ts", "tokens": "$close d:20 Std", "tags": "ts,volatility", "family": "ts", "source": "builtin_v1"},
    {"name": "ema_close_20", "head": "ts", "tokens": "$close d:20 EMA", "tags": "ts,ema", "family": "ts", "source": "builtin_v1"},
    {"name": "wma_close_20", "head": "ts", "tokens": "$close d:20 WMA", "tags": "ts,wma", "family": "ts", "source": "builtin_v1"},
    {"name": "corr_close_volume_20", "head": "ts", "tokens": "$close $volume d:20 Corr", "tags": "ts,corr", "family": "ts", "source": "builtin_v1"},
    {"name": "cov_high_low_20", "head": "ts", "tokens": "$high $low d:20 Cov", "tags": "ts,cov", "family": "ts", "source": "builtin_v1"},
    {"name": "delta_vwap_20", "head": "ts", "tokens": "$vwap d:20 Delta", "tags": "ts,momentum", "family": "ts", "source": "builtin_v1"},
    {"name": "mean_volume_20", "head": "ts", "tokens": "$volume d:20 Mean", "tags": "ts,volume", "family": "ts", "source": "builtin_v1"},
    {"name": "price_volume_corr", "head": "pv", "tokens": "$close $volume d:20 Corr", "tags": "pv,corr", "family": "pv", "source": "builtin_v1"},
    {"name": "vwap_volume_corr", "head": "pv", "tokens": "$vwap $volume d:20 Corr", "tags": "pv,corr", "family": "pv", "source": "builtin_v1"},
    {"name": "price_volume_momentum", "head": "pv", "tokens": "$close d:20 Delta $volume d:20 Delta Mul", "tags": "pv,momentum", "family": "pv", "source": "builtin_v1"},
    {"name": "volume_adjusted_return", "head": "pv", "tokens": "$close $open Sub $volume Log Div", "tags": "pv,return", "family": "pv", "source": "builtin_v1"},
    {"name": "money_flow_mean_20", "head": "pv", "tokens": "$close $volume Mul d:20 Mean", "tags": "pv,moneyflow", "family": "pv", "source": "builtin_v1"},
    {"name": "vwap_volume_mean_20", "head": "pv", "tokens": "$vwap $volume Mul d:20 Mean", "tags": "pv,moneyflow", "family": "pv", "source": "builtin_v1"},
    {"name": "close_above_mean_20", "head": "rank", "tokens": "$close $close d:20 Mean Greater", "tags": "rank,relative", "family": "relative", "source": "builtin_v1"},
    {"name": "close_below_mean_20", "head": "rank", "tokens": "$close $close d:20 Mean Less", "tags": "rank,relative", "family": "relative", "source": "builtin_v1"},
    {"name": "volume_above_mean_20", "head": "rank", "tokens": "$volume $volume d:20 Mean Greater", "tags": "rank,volume", "family": "relative", "source": "builtin_v1"},
    {"name": "vwap_above_close", "head": "rank", "tokens": "$vwap $close Greater", "tags": "rank,price", "family": "relative", "source": "builtin_v1"},
    {"name": "high_breakout_20", "head": "rank", "tokens": "$high $high d:20 Max Greater", "tags": "rank,breakout", "family": "relative", "source": "builtin_v1"},
    {"name": "low_breakdown_20", "head": "rank", "tokens": "$low $low d:20 Min Less", "tags": "rank,breakout", "family": "relative", "source": "builtin_v1"},
    {"name": "log_abs_vwap_gap", "head": "explore", "tokens": "$vwap $close Sub Abs Log", "tags": "explore,mixed", "family": "explore", "source": "builtin_v1"},
    {"name": "mean_high_low_spread_20", "head": "explore", "tokens": "$high $low Sub d:20 Mean", "tags": "explore,range", "family": "explore", "source": "builtin_v1"},
    {"name": "mad_volume_20", "head": "explore", "tokens": "$volume d:20 Mad", "tags": "explore,volume", "family": "explore", "source": "builtin_v1"},
    {"name": "med_vwap_30", "head": "explore", "tokens": "$vwap d:30 Med", "tags": "explore,median", "family": "explore", "source": "builtin_v1"},
    {"name": "corr_high_volume_30", "head": "explore", "tokens": "$high $volume d:30 Corr", "tags": "explore,corr", "family": "explore", "source": "builtin_v1"},
    {"name": "abs_delta_low_20", "head": "explore", "tokens": "$low d:20 Delta Abs", "tags": "explore,momentum", "family": "explore", "source": "builtin_v1"},
]


STRONG_EXTRA_FACTORS: List[Dict[str, str]] = [
    {"name": "alpha101_proxy_intraday_reversal", "head": "base", "tokens": "$open $close Sub $high $low Sub Div", "tags": "alpha101,range,reversal", "family": "alpha101", "source": "alpha101_proxy", "weight": "1.5"},
    {"name": "alpha101_proxy_vwap_reversal", "head": "base", "tokens": "$vwap $close Sub $high $low Sub Div", "tags": "alpha101,vwap,reversal", "family": "alpha101", "source": "alpha101_proxy", "weight": "1.5"},
    {"name": "alpha101_proxy_gap_reversal", "head": "base", "tokens": "$open $close d:10 Ref Sub $close Div", "tags": "alpha101,gap", "family": "alpha101", "source": "alpha101_proxy"},
    {"name": "alpha101_proxy_range_position", "head": "base", "tokens": "$close $low Sub $high $low Sub Div", "tags": "alpha101,range", "family": "alpha101", "source": "alpha101_proxy", "weight": "1.5"},
    {"name": "alpha101_proxy_vwap_position", "head": "base", "tokens": "$close $vwap Sub $high $low Sub Div", "tags": "alpha101,vwap", "family": "alpha101", "source": "alpha101_proxy"},
    {"name": "alpha158_kbar", "head": "simple", "tokens": "$close $open Sub $open Div", "tags": "alpha158,kbar", "family": "alpha158", "source": "alpha158_proxy", "weight": "1.5"},
    {"name": "alpha158_upper_shadow_proxy", "head": "simple", "tokens": "$high $close Sub Abs $open Div", "tags": "alpha158,kline", "family": "alpha158", "source": "alpha158_proxy"},
    {"name": "alpha158_lower_shadow_proxy", "head": "simple", "tokens": "$close $low Sub Abs $open Div", "tags": "alpha158,kline", "family": "alpha158", "source": "alpha158_proxy"},
    {"name": "alpha158_price_range", "head": "simple", "tokens": "$high $low Sub $open Div", "tags": "alpha158,range", "family": "alpha158", "source": "alpha158_proxy"},
    {"name": "alpha158_vwap_gap", "head": "simple", "tokens": "$close $vwap Sub $vwap Div", "tags": "alpha158,vwap", "family": "alpha158", "source": "alpha158_proxy"},
    {"name": "simple_close_vwap_abs_gap", "head": "simple", "tokens": "$close $vwap Sub Abs", "tags": "simple,vwap", "family": "simple", "source": "formulaic"},
    {"name": "simple_open_vwap_gap", "head": "simple", "tokens": "$open $vwap Sub", "tags": "simple,vwap", "family": "simple", "source": "formulaic"},
    {"name": "simple_high_close_gap", "head": "simple", "tokens": "$high $close Sub", "tags": "simple,range", "family": "simple", "source": "formulaic"},
    {"name": "simple_close_low_gap", "head": "simple", "tokens": "$close $low Sub", "tags": "simple,range", "family": "simple", "source": "formulaic"},
    {"name": "simple_mid_close_gap", "head": "simple", "tokens": "$high $low Add c:2.0 Div $close Sub", "tags": "simple,range", "family": "simple", "source": "formulaic"},
    {"name": "simple_intraday_abs_return", "head": "simple", "tokens": "$close $open Sub Abs $open Div", "tags": "simple,return", "family": "simple", "source": "formulaic"},
    {"name": "ts_delta_close_10", "head": "ts", "tokens": "$close d:10 Delta $close Div", "tags": "ts,momentum", "family": "ts", "source": "formulaic"},
    {"name": "ts_delta_vwap_10", "head": "ts", "tokens": "$vwap d:10 Delta $vwap Div", "tags": "ts,momentum,vwap", "family": "ts", "source": "formulaic"},
    {"name": "ts_ref_return_10", "head": "ts", "tokens": "$close $close d:10 Ref Sub $close d:10 Ref Div", "tags": "ts,return", "family": "ts", "source": "formulaic"},
    {"name": "ts_ma_gap_20", "head": "ts", "tokens": "$close $close d:20 Mean Sub $close d:20 Std Div", "tags": "ts,zscore", "family": "ts", "source": "formulaic", "weight": "1.5"},
    {"name": "ts_vwap_ma_gap_20", "head": "ts", "tokens": "$vwap $vwap d:20 Mean Sub $vwap d:20 Std Div", "tags": "ts,zscore,vwap", "family": "ts", "source": "formulaic"},
    {"name": "ts_volume_zscore_20", "head": "ts", "tokens": "$volume $volume d:20 Mean Sub $volume d:20 Std Div", "tags": "ts,volume,zscore", "family": "ts", "source": "formulaic"},
    {"name": "ts_price_volatility_ratio", "head": "ts", "tokens": "$close d:20 Std $close d:50 Std Div", "tags": "ts,volatility", "family": "ts", "source": "formulaic"},
    {"name": "ts_range_volatility_20", "head": "ts", "tokens": "$high $low Sub d:20 Std", "tags": "ts,range,volatility", "family": "ts", "source": "formulaic"},
    {"name": "ts_corr_close_volume_10", "head": "ts", "tokens": "$close $volume d:10 Corr", "tags": "ts,corr,pv", "family": "ts", "source": "formulaic"},
    {"name": "ts_rank_close_20", "head": "ts", "tokens": "$close d:20 TSRank", "tags": "ts,rank,worldquant", "family": "ts", "source": "worldquant_proxy", "weight": "1.5"},
    {"name": "ts_rank_vwap_20", "head": "ts", "tokens": "$vwap d:20 TSRank", "tags": "ts,rank,vwap", "family": "ts", "source": "worldquant_proxy", "weight": "1.5"},
    {"name": "ts_rank_volume_20", "head": "ts", "tokens": "$volume d:20 TSRank", "tags": "ts,rank,volume", "family": "ts", "source": "worldquant_proxy"},
    {"name": "ts_rank_return_20", "head": "ts", "tokens": "$close d:20 Delta d:20 TSRank", "tags": "ts,rank,momentum", "family": "ts", "source": "worldquant_proxy"},
    {"name": "pv_money_flow_delta_20", "head": "pv", "tokens": "$close $volume Mul d:20 Delta", "tags": "pv,moneyflow", "family": "pv", "source": "formulaic", "weight": "1.5"},
    {"name": "pv_vwap_money_flow_delta_20", "head": "pv", "tokens": "$vwap $volume Mul d:20 Delta", "tags": "pv,moneyflow,vwap", "family": "pv", "source": "formulaic"},
    {"name": "pv_volume_weighted_momentum", "head": "pv", "tokens": "$close d:20 Delta $volume d:20 Mean Mul", "tags": "pv,momentum", "family": "pv", "source": "formulaic"},
    {"name": "pv_vwap_volume_momentum", "head": "pv", "tokens": "$vwap d:20 Delta $volume d:20 Mean Mul", "tags": "pv,momentum,vwap", "family": "pv", "source": "formulaic"},
    {"name": "pv_corr_return_volume", "head": "pv", "tokens": "$close d:20 Delta $volume d:20 Delta d:20 Corr", "tags": "pv,corr", "family": "pv", "source": "formulaic"},
    {"name": "pv_volume_shock_return", "head": "pv", "tokens": "$volume $volume d:20 Mean Div $close d:20 Delta Mul", "tags": "pv,shock", "family": "pv", "source": "formulaic"},
    {"name": "pv_turnover_proxy", "head": "pv", "tokens": "$volume d:10 Mean $volume d:50 Mean Div", "tags": "pv,volume", "family": "pv", "source": "formulaic"},
    {"name": "relative_close_vs_mean", "head": "rank", "tokens": "$close $close d:20 Mean Greater", "tags": "relative,compare", "family": "relative", "source": "formulaic", "weight": "1.5"},
    {"name": "relative_close_vs_vwap", "head": "rank", "tokens": "$close $vwap Greater", "tags": "relative,vwap", "family": "relative", "source": "formulaic"},
    {"name": "relative_vwap_vs_mean", "head": "rank", "tokens": "$vwap $vwap d:20 Mean Greater", "tags": "relative,vwap", "family": "relative", "source": "formulaic"},
    {"name": "relative_volume_breakout", "head": "rank", "tokens": "$volume $volume d:20 Max Greater", "tags": "relative,volume,breakout", "family": "relative", "source": "formulaic"},
    {"name": "relative_close_drawdown", "head": "rank", "tokens": "$close $close d:20 Max Less", "tags": "relative,drawdown", "family": "relative", "source": "formulaic"},
    {"name": "relative_low_breakdown", "head": "rank", "tokens": "$low $low d:20 Min Less", "tags": "relative,breakdown", "family": "relative", "source": "formulaic"},
    {"name": "relative_high_breakout", "head": "rank", "tokens": "$high $high d:20 Max Greater", "tags": "relative,breakout", "family": "relative", "source": "formulaic"},
    {"name": "csrank_close", "head": "rank", "tokens": "$close CSRank", "tags": "rank,csrank,worldquant", "family": "relative", "source": "worldquant_proxy", "weight": "2.0"},
    {"name": "csrank_vwap_gap", "head": "rank", "tokens": "$vwap $close Sub CSRank", "tags": "rank,csrank,vwap", "family": "relative", "source": "worldquant_proxy", "weight": "1.5"},
    {"name": "csrank_intraday_return", "head": "rank", "tokens": "$close $open Sub $open Div CSRank", "tags": "rank,csrank,return", "family": "relative", "source": "worldquant_proxy", "weight": "1.5"},
    {"name": "csrank_volume", "head": "rank", "tokens": "$volume CSRank", "tags": "rank,csrank,volume", "family": "relative", "source": "worldquant_proxy"},
    {"name": "csrank_range_position", "head": "rank", "tokens": "$close $low Sub $high $low Sub Div CSRank", "tags": "rank,csrank,range", "family": "relative", "source": "worldquant_proxy", "weight": "1.5"},
    {"name": "signed_close_momentum", "head": "rank", "tokens": "$close d:20 Delta Sign", "tags": "sign,momentum", "family": "relative", "source": "worldquant_proxy"},
    {"name": "signed_volume_shock", "head": "rank", "tokens": "$volume $volume d:20 Mean Sub Sign", "tags": "sign,volume", "family": "relative", "source": "worldquant_proxy"},
    {"name": "explore_abs_range_log", "head": "explore", "tokens": "$high $low Sub Abs Log", "tags": "explore,range", "family": "explore", "source": "formulaic"},
    {"name": "explore_corr_range_volume", "head": "explore", "tokens": "$high $low Sub $volume d:20 Corr", "tags": "explore,corr,pv", "family": "explore", "source": "formulaic"},
    {"name": "explore_mean_abs_gap", "head": "explore", "tokens": "$close $open Sub Abs d:20 Mean", "tags": "explore,gap", "family": "explore", "source": "formulaic"},
    {"name": "explore_mad_vwap_volume", "head": "explore", "tokens": "$vwap $volume Mul d:20 Mad", "tags": "explore,pv,mad", "family": "explore", "source": "formulaic"},
    {"name": "explore_price_volume_cov", "head": "explore", "tokens": "$close $volume d:20 Cov", "tags": "explore,pv,cov", "family": "explore", "source": "formulaic"},
    {"name": "explore_safe_sqrt_range", "head": "explore", "tokens": "$high $low Sub SafeSqrt", "tags": "explore,sqrt,range", "family": "explore", "source": "worldquant_proxy"},
    {"name": "explore_sqrt_volume", "head": "explore", "tokens": "$volume SafeSqrt", "tags": "explore,sqrt,volume", "family": "explore", "source": "worldquant_proxy"},
]


UNSUPPORTED_ALPHA101_FACTORS: List[Dict[str, str]] = [
    {"name": "alpha101_002_original", "head": "rank", "family": "alpha101", "source": "alpha101_original", "skip_reason": "requires nested Rank(Corr(Rank(...), Rank(...))) with original alpha constants; proxied by csrank/tsrank templates"},
    {"name": "alpha101_006_original", "head": "rank", "family": "alpha101", "source": "alpha101_original", "skip_reason": "proxied by CSRank templates; original formula still needs exact WorldQuant preprocessing"},
    {"name": "alpha101_007_original", "head": "rank", "family": "alpha101", "source": "alpha101_original", "skip_reason": "requires adv20 conditional expression and Rank operator"},
    {"name": "alpha101_021_original", "head": "rank", "family": "alpha101", "source": "alpha101_original", "skip_reason": "requires ternary conditional and Sum comparison"},
    {"name": "alpha101_025_original", "head": "rank", "family": "alpha101", "source": "alpha101_original", "skip_reason": "requires Rank and signed power style transform"},
    {"name": "alpha101_031_original", "head": "rank", "family": "alpha101", "source": "alpha101_original", "skip_reason": "requires DecayLinear and Rank operators"},
    {"name": "alpha101_041_original", "head": "rank", "family": "alpha101", "source": "alpha101_original", "skip_reason": "proxied by SafeSqrt templates; original formula requires exact signed rank-style expression"},
    {"name": "alpha101_054_original", "head": "rank", "family": "alpha101", "source": "alpha101_original", "skip_reason": "requires power operator and unsupported constants"},
    {"name": "alpha101_101_original", "head": "simple", "family": "alpha101", "source": "alpha101_original", "skip_reason": "available as alpha101_proxy_intraday_reversal under current token set"},
]


# Backward-compatible alias used by older imports.
BUILTIN_CLASSIC_FACTORS = BUILTIN_V1_CLASSIC_FACTORS


def _action_id_for_token(token: Token) -> int:
    if isinstance(token, OperatorToken):
        return OFFSET_OP + OPERATORS.index(token.operator) - 1
    if isinstance(token, FeatureToken):
        return OFFSET_FEATURE + int(token.feature) - 1
    if isinstance(token, DeltaTimeToken):
        return OFFSET_DELTA_TIME + DELTA_TIMES.index(token.delta_time) - 1
    if isinstance(token, ConstantToken):
        return OFFSET_CONSTANT + CONSTANTS.index(float(token.constant)) - 1
    raise ValueError(f"unsupported token: {token}")


def _parse_numeric(raw: str) -> float:
    constant_match = CONSTANT_PATTERN.match(raw)
    if constant_match:
        return float(constant_match.group(1))
    if NUMBER_PATTERN.match(raw):
        return float(raw)
    raise ValueError(f"not a numeric token: {raw}")


def _token_from_text(raw: str, builder: ExpressionBuilder, next_raw: Optional[str]) -> Token:
    if raw.startswith("$"):
        return FeatureToken(FeatureType[raw[1:].upper()])
    if raw in OP_BY_NAME:
        return OperatorToken(OP_BY_NAME[raw])

    lower = raw.lower()
    if lower.startswith("d:"):
        dt = int(float(raw[2:]))
        if dt not in DELTA_TIMES:
            raise ValueError(f"delta time not supported: {raw}")
        return DeltaTimeToken(dt)
    if lower.startswith("c:"):
        value = float(raw[2:])
        if value not in CONSTANTS:
            raise ValueError(f"constant not supported: {raw}")
        return ConstantToken(value)

    value = _parse_numeric(raw)
    next_op = OP_BY_NAME.get(next_raw or "")
    if (
        next_op is not None
        and issubclass(next_op, (RollingOperator, PairRollingOperator))
        and int(value) == value
        and int(value) in DELTA_TIMES
        and builder.validate_dt()
    ):
        return DeltaTimeToken(int(value))
    if value not in CONSTANTS:
        raise ValueError(f"constant not supported: {raw}; use one of {CONSTANTS}")
    return ConstantToken(value)


def parse_rpn_tokens(token_text: Iterable[str]) -> List[int]:
    raw_tokens = [x.strip() for x in token_text if x.strip()]
    if raw_tokens and raw_tokens[-1].upper() == "SEP":
        raw_tokens = raw_tokens[:-1]
    if len(raw_tokens) > MAX_EXPR_LENGTH:
        raise ValueError(f"expression exceeds MAX_EXPR_LENGTH={MAX_EXPR_LENGTH}")
    builder = ExpressionBuilder()
    action_ids: List[int] = []
    for idx, raw in enumerate(raw_tokens):
        next_raw = raw_tokens[idx + 1] if idx + 1 < len(raw_tokens) else None
        token = _token_from_text(raw, builder, next_raw)
        builder.add_token(token)
        action_ids.append(_action_id_for_token(token))
    builder.get_tree()
    if not builder.is_valid():
        raise ValueError("expression is not a valid featured expression")
    return action_ids


def _record_defaults(record: Dict[str, str], fallback_source: str) -> Dict[str, str]:
    out = dict(record)
    out.setdefault("family", out.get("head", ""))
    out.setdefault("source", fallback_source)
    out.setdefault("tags", out.get("family", ""))
    out.setdefault("weight", "1.0")
    out.setdefault("comment", "")
    out["head"] = _infer_head(out)
    return out


def _parse_factor_record(record: Dict[str, str]) -> Tuple[Optional[ClassicFactor], Optional[SkippedFactor]]:
    name = (record.get("name") or "").strip()
    head = (record.get("head") or "").strip()
    family = (record.get("family") or head).strip()
    source = (record.get("source") or "").strip()
    if not name or name.startswith("#"):
        return None, None
    if record.get("skip_reason"):
        return None, SkippedFactor(name=name, head=head, family=family, source=source, reason=str(record["skip_reason"]))
    tokens = (record.get("tokens") or "").strip()
    tags = tuple(x.strip() for x in (record.get("tags") or "").split(",") if x.strip())
    if head not in HEAD_SET:
        return None, SkippedFactor(name=name, head=head, family=family, source=source, reason=f"unsupported head: {head}")
    if not tokens:
        return None, SkippedFactor(name=name, head=head, family=family, source=source, reason="empty tokens")
    try:
        weight = float(record.get("weight") or 1.0)
    except ValueError:
        return None, SkippedFactor(name=name, head=head, family=family, source=source, reason=f"invalid weight: {record.get('weight')}")
    token_text = tokens.split()
    try:
        action_ids = parse_rpn_tokens(token_text)
    except Exception as exc:
        return None, SkippedFactor(name=name, head=head, family=family, source=source, reason=str(exc))
    return (
        ClassicFactor(
            name=name,
            head=head,
            token_text=token_text,
            action_ids=action_ids,
            tags=tags,
            family=family,
            weight=weight,
            source=source,
            comment=(record.get("comment") or "").strip(),
            augmented_from=(record.get("augmented_from") or "").strip(),
        ),
        None,
    )


def _load_csv_records(csv_path: str) -> List[Dict[str, str]]:
    if not csv_path:
        return []
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"classic factor csv not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "head", "tokens"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"classic factor csv missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def _base_records_for_bank(bank: str) -> List[Dict[str, str]]:
    if bank == "builtin_v1":
        return [_record_defaults(record, "builtin_v1") for record in BUILTIN_V1_CLASSIC_FACTORS]
    if bank != "strong":
        raise ValueError(f"unsupported classic factor bank: {bank}")
    records = [_record_defaults(record, "builtin_v1") for record in BUILTIN_V1_CLASSIC_FACTORS]
    records.extend(_record_defaults(record, "strong_builtin") for record in STRONG_EXTRA_FACTORS)
    records.extend(_record_defaults(record, "alpha101_original") for record in UNSUPPORTED_ALPHA101_FACTORS)
    return records


def _augment_record(record: Dict[str, str], name_suffix: str, tokens: str, extra_tags: str) -> Dict[str, str]:
    out = dict(record)
    out["name"] = f"{record['name']}__{name_suffix}"
    out["tokens"] = tokens
    out["tags"] = ",".join(x for x in [record.get("tags", ""), "augmented", extra_tags] if x)
    out["source"] = f"{record.get('source', '')}:augmented"
    out["augmented_from"] = record["name"]
    return out


def _augment_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    augmented: List[Dict[str, str]] = []
    for record in records:
        augmented.append(record)
        if record.get("skip_reason"):
            continue
        tokens = record.get("tokens", "")
        family = record.get("family", "")
        tags = record.get("tags", "")
        if "d:20" in tokens:
            for window in (10, 30, 40, 50):
                augmented.append(_augment_record(record, f"w{window}", tokens.replace("d:20", f"d:{window}"), f"window:{window}"))
        if "$close" in tokens and "$volume" not in tokens and family in {"simple", "ts", "alpha101", "alpha158"}:
            for field in ("$vwap", "$open"):
                field_name = field[1:]
                augmented.append(_augment_record(record, f"{field_name}", tokens.replace("$close", field), f"field:{field_name}"))
        if family in {"pv", "alpha101", "alpha158"} and "$volume" not in tokens and "$close" in tokens:
            pv_tokens = f"{tokens} $volume d:20 Mean Mul"
            augmented.append(_augment_record(record, "pv_weighted", pv_tokens, "pv_weighted"))
    return augmented


def _dedupe_factors(factors: List[ClassicFactor]) -> Tuple[List[ClassicFactor], List[SkippedFactor]]:
    deduped: List[ClassicFactor] = []
    skipped: List[SkippedFactor] = []
    seen = set()
    for factor in factors:
        key = tuple(factor.action_ids)
        if key in seen:
            skipped.append(
                SkippedFactor(
                    name=factor.name,
                    head=factor.head,
                    family=factor.family,
                    source=factor.source,
                    reason="duplicate token sequence",
                )
            )
            continue
        seen.add(key)
        deduped.append(factor)
    return deduped, skipped


def load_classic_factor_bank(
    csv_path: str = "",
    bank: str = "strong",
    augment: bool = True,
) -> ClassicFactorLoadResult:
    base_records = _base_records_for_bank(bank)
    csv_records = [_record_defaults(record, "csv") for record in _load_csv_records(csv_path)]
    raw_records = base_records + csv_records
    candidate_records = _augment_records(raw_records) if augment and bank == "strong" else raw_records

    parsed: List[ClassicFactor] = []
    skipped: List[SkippedFactor] = []
    for record in candidate_records:
        factor, error = _parse_factor_record(record)
        if error is not None:
            skipped.append(error)
            continue
        if factor is not None:
            parsed.append(factor)
    deduped, duplicate_skips = _dedupe_factors(parsed)
    skipped.extend(duplicate_skips)

    stats = {
        "bank": bank,
        "augment_enabled": bool(augment and bank == "strong"),
        "builtin_v1_record_count": len(BUILTIN_V1_CLASSIC_FACTORS),
        "strong_extra_record_count": len(STRONG_EXTRA_FACTORS) if bank == "strong" else 0,
        "unsupported_record_count": len(UNSUPPORTED_ALPHA101_FACTORS) if bank == "strong" else 0,
        "csv_record_count": len(csv_records),
        "raw_record_count": len(raw_records),
        "candidate_record_count": len(candidate_records),
        "parsed_before_dedupe_count": len(parsed),
        "loaded_count": len(deduped),
        "skipped_count": len(skipped),
        "deduped_count": len(duplicate_skips),
        "augmented_candidate_count": max(0, len(candidate_records) - len(raw_records)),
        "head_counts": dict(Counter(f.head for f in deduped)),
        "family_counts": dict(Counter(f.family for f in deduped)),
        "source_counts": dict(Counter(f.source for f in deduped)),
    }
    return ClassicFactorLoadResult(factors=deduped, skipped=skipped, stats=stats)


def load_classic_factors(
    csv_path: str = "",
    bank: str = "builtin_v1",
    augment: bool = False,
) -> Tuple[List[ClassicFactor], List[SkippedFactor]]:
    result = load_classic_factor_bank(csv_path=csv_path, bank=bank, augment=augment)
    return result.factors, result.skipped
