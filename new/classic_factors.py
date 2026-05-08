import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from alphagen.config import CONSTANTS, DELTA_TIMES, OPERATORS
from alphagen.data.expression import BinaryOperator, PairRollingOperator, RollingOperator, UnaryOperator
from alphagen.data.tokens import ConstantToken, DeltaTimeToken, FeatureToken, OperatorToken, Token
from alphagen.data.tree import ExpressionBuilder
from alphagen.rl.env.wrapper import OFFSET_CONSTANT, OFFSET_DELTA_TIME, OFFSET_FEATURE, OFFSET_OP
from alphagen_qlib.stock_data import FeatureType
from new.env import HEAD_NAMES


NUMBER_PATTERN = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?$", re.IGNORECASE)
CONSTANT_PATTERN = re.compile(r"^Constant\(([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\)$", re.IGNORECASE)

OP_BY_NAME = {op.__name__: op for op in OPERATORS}
HEAD_SET = set(HEAD_NAMES)


@dataclass(frozen=True)
class ClassicFactor:
    name: str
    head: str
    token_text: List[str]
    action_ids: List[int]
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SkippedFactor:
    name: str
    head: str
    reason: str


BUILTIN_CLASSIC_FACTORS: List[Dict[str, str]] = [
    {"name": "close_open_return", "head": "base", "tokens": "$close $open Sub $open Div", "tags": "return,simple"},
    {"name": "high_low_spread", "head": "base", "tokens": "$high $low Sub $close Div", "tags": "range"},
    {"name": "vwap_close_gap", "head": "base", "tokens": "$vwap $close Sub $close Div", "tags": "price"},
    {"name": "price_momentum_20", "head": "base", "tokens": "$close d:20 Delta $close Div", "tags": "momentum"},
    {"name": "intraday_position", "head": "base", "tokens": "$close $low Sub $high $low Sub Div", "tags": "range"},
    {"name": "volume_momentum_20", "head": "base", "tokens": "$volume d:20 Delta $volume Div", "tags": "volume"},
    {"name": "close_minus_open", "head": "simple", "tokens": "$close $open Sub", "tags": "simple"},
    {"name": "high_minus_low", "head": "simple", "tokens": "$high $low Sub", "tags": "simple"},
    {"name": "close_div_vwap", "head": "simple", "tokens": "$close $vwap Div", "tags": "simple"},
    {"name": "abs_intraday_return", "head": "simple", "tokens": "$close $open Sub Abs", "tags": "simple"},
    {"name": "log_volume", "head": "simple", "tokens": "$volume Log", "tags": "simple,volume"},
    {"name": "close_low_gap", "head": "simple", "tokens": "$close $low Sub", "tags": "simple"},
    {"name": "mean_close_20", "head": "ts", "tokens": "$close d:20 Mean", "tags": "ts,mean"},
    {"name": "std_close_20", "head": "ts", "tokens": "$close d:20 Std", "tags": "ts,volatility"},
    {"name": "ema_close_20", "head": "ts", "tokens": "$close d:20 EMA", "tags": "ts,ema"},
    {"name": "wma_close_20", "head": "ts", "tokens": "$close d:20 WMA", "tags": "ts,wma"},
    {"name": "corr_close_volume_20", "head": "ts", "tokens": "$close $volume d:20 Corr", "tags": "ts,corr"},
    {"name": "cov_high_low_20", "head": "ts", "tokens": "$high $low d:20 Cov", "tags": "ts,cov"},
    {"name": "delta_vwap_20", "head": "ts", "tokens": "$vwap d:20 Delta", "tags": "ts,momentum"},
    {"name": "mean_volume_20", "head": "ts", "tokens": "$volume d:20 Mean", "tags": "ts,volume"},
    {"name": "price_volume_corr", "head": "pv", "tokens": "$close $volume d:20 Corr", "tags": "pv,corr"},
    {"name": "vwap_volume_corr", "head": "pv", "tokens": "$vwap $volume d:20 Corr", "tags": "pv,corr"},
    {"name": "price_volume_momentum", "head": "pv", "tokens": "$close d:20 Delta $volume d:20 Delta Mul", "tags": "pv,momentum"},
    {"name": "volume_adjusted_return", "head": "pv", "tokens": "$close $open Sub $volume Log Div", "tags": "pv,return"},
    {"name": "money_flow_mean_20", "head": "pv", "tokens": "$close $volume Mul d:20 Mean", "tags": "pv,moneyflow"},
    {"name": "vwap_volume_mean_20", "head": "pv", "tokens": "$vwap $volume Mul d:20 Mean", "tags": "pv,moneyflow"},
    {"name": "close_above_mean_20", "head": "rank", "tokens": "$close $close d:20 Mean Greater", "tags": "rank,relative"},
    {"name": "close_below_mean_20", "head": "rank", "tokens": "$close $close d:20 Mean Less", "tags": "rank,relative"},
    {"name": "volume_above_mean_20", "head": "rank", "tokens": "$volume $volume d:20 Mean Greater", "tags": "rank,volume"},
    {"name": "vwap_above_close", "head": "rank", "tokens": "$vwap $close Greater", "tags": "rank,price"},
    {"name": "high_breakout_20", "head": "rank", "tokens": "$high $high d:20 Max Greater", "tags": "rank,breakout"},
    {"name": "low_breakdown_20", "head": "rank", "tokens": "$low $low d:20 Min Less", "tags": "rank,breakout"},
    {"name": "log_abs_vwap_gap", "head": "explore", "tokens": "$vwap $close Sub Abs Log", "tags": "explore,mixed"},
    {"name": "mean_high_low_spread_20", "head": "explore", "tokens": "$high $low Sub d:20 Mean", "tags": "explore,range"},
    {"name": "mad_volume_20", "head": "explore", "tokens": "$volume d:20 Mad", "tags": "explore,volume"},
    {"name": "med_vwap_30", "head": "explore", "tokens": "$vwap d:30 Med", "tags": "explore,median"},
    {"name": "corr_high_volume_30", "head": "explore", "tokens": "$high $volume d:30 Corr", "tags": "explore,corr"},
    {"name": "abs_delta_low_20", "head": "explore", "tokens": "$low d:20 Delta Abs", "tags": "explore,momentum"},
]


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


def _parse_factor_record(record: Dict[str, str]) -> Tuple[Optional[ClassicFactor], Optional[SkippedFactor]]:
    name = (record.get("name") or "").strip()
    head = (record.get("head") or "").strip()
    tokens = (record.get("tokens") or "").strip()
    tags = tuple(x.strip() for x in (record.get("tags") or "").split(",") if x.strip())
    if not name or name.startswith("#"):
        return None, None
    if head not in HEAD_SET:
        return None, SkippedFactor(name=name, head=head, reason=f"unsupported head: {head}")
    if not tokens:
        return None, SkippedFactor(name=name, head=head, reason="empty tokens")
    token_text = tokens.split()
    try:
        action_ids = parse_rpn_tokens(token_text)
    except Exception as exc:
        return None, SkippedFactor(name=name, head=head, reason=str(exc))
    return ClassicFactor(name=name, head=head, token_text=token_text, action_ids=action_ids, tags=tags), None


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


def load_classic_factors(csv_path: str = "") -> Tuple[List[ClassicFactor], List[SkippedFactor]]:
    records = list(BUILTIN_CLASSIC_FACTORS) + _load_csv_records(csv_path)
    factors: List[ClassicFactor] = []
    skipped: List[SkippedFactor] = []
    seen = set()
    for record in records:
        factor, error = _parse_factor_record(record)
        if error is not None:
            skipped.append(error)
            continue
        if factor is None:
            continue
        key = (factor.name, factor.head)
        if key in seen:
            skipped.append(SkippedFactor(name=factor.name, head=factor.head, reason="duplicate factor"))
            continue
        seen.add(key)
        factors.append(factor)
    return factors, skipped
