from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor

from alphagen.data.expression import (
    BinaryOperator,
    Constant,
    Corr,
    Cov,
    Div,
    Expression,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)
from alphagen.utils.correlation import batch_spearmanr


RISKY_OPERATOR_TYPES = (Div, Corr, Cov)


def _safe_std(x: Tensor, dim: int) -> Tensor:
    return x.std(dim=dim, unbiased=False).clamp_min(1e-6)


def long_short_returns(signal: Tensor, target: Tensor, top_frac: float = 0.2) -> Tensor:
    if signal.ndim != 2 or target.ndim != 2:
        raise ValueError("signal and target must be [days, stocks]")
    days, stocks = signal.shape
    k = max(1, int(stocks * top_frac))
    daily_ret = torch.zeros(days, dtype=signal.dtype, device=signal.device)
    for day in range(days):
        sig = signal[day]
        tgt = target[day]
        top_idx = torch.topk(sig, k=k, largest=True).indices
        bottom_idx = torch.topk(sig, k=k, largest=False).indices
        daily_ret[day] = tgt[top_idx].mean() - tgt[bottom_idx].mean()
    return daily_ret


def signal_turnover(signal: Tensor, top_frac: float = 0.2) -> float:
    if signal.ndim != 2 or signal.shape[0] < 2:
        return 0.0
    days, stocks = signal.shape
    k = max(1, int(stocks * top_frac))
    positions = torch.zeros_like(signal)
    for day in range(days):
        sig = signal[day]
        top_idx = torch.topk(sig, k=k, largest=True).indices
        bottom_idx = torch.topk(sig, k=k, largest=False).indices
        positions[day, top_idx] = 1.0
        positions[day, bottom_idx] = -1.0
    diffs = (positions[1:] - positions[:-1]).abs().mean(dim=1)
    return float(diffs.mean().item()) if diffs.numel() else 0.0


def rankic(signal: Tensor, target: Tensor) -> float:
    return float(batch_spearmanr(signal, target).mean().item())


def combined_metric(
    signal: Tensor,
    target: Tensor,
    eta: float = 0.0,
    xi: float = 0.0,
    top_frac: float = 0.2,
) -> dict[str, float]:
    metric_rankic = rankic(signal, target)
    daily_returns = long_short_returns(signal, target, top_frac=top_frac)
    daily_mean = float(daily_returns.mean().item())
    daily_std = float(_safe_std(daily_returns, dim=0).item())
    sharpe = 0.0 if math.isclose(daily_std, 0.0) else daily_mean / daily_std * math.sqrt(252.0)
    turnover = signal_turnover(signal, top_frac=top_frac)
    metric = metric_rankic + eta * sharpe - xi * turnover
    return {
        "metric": float(metric),
        "rankic": float(metric_rankic),
        "sharpe": float(sharpe),
        "turnover": float(turnover),
    }


def iter_children(expr: Expression) -> Iterable[Expression]:
    if isinstance(expr, UnaryOperator):
        yield expr._operand
    elif isinstance(expr, BinaryOperator):
        yield expr._lhs
        yield expr._rhs
    elif isinstance(expr, RollingOperator):
        yield expr._operand
        yield Constant(float(expr._delta_time))
    elif isinstance(expr, PairRollingOperator):
        yield expr._lhs
        yield expr._rhs
        yield Constant(float(expr._delta_time))


def expression_length(expr: Expression) -> int:
    return 1 + sum(expression_length(child) for child in iter_children(expr))


def expression_depth(expr: Expression) -> int:
    children = list(iter_children(expr))
    if not children:
        return 1
    return 1 + max(expression_depth(child) for child in children)


def risky_operator_count(expr: Expression) -> int:
    current = 1 if isinstance(expr, RISKY_OPERATOR_TYPES) else 0
    return current + sum(risky_operator_count(child) for child in iter_children(expr))


def flatten_valid(x: Tensor) -> Tensor:
    return torch.nan_to_num(x.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def stack_alpha_panels(alpha_panels: Sequence[Tensor]) -> Tensor:
    if not alpha_panels:
        raise ValueError("alpha_panels must not be empty")
    return torch.stack(list(alpha_panels), dim=0)


def weight_sparsity(weights: Tensor, eps: float = 1e-4) -> float:
    if weights.numel() == 0:
        return 0.0
    return float((weights.abs() < eps).float().mean().item())
