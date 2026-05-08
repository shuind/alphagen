import re
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn

from alphagen.config import MAX_EXPR_LENGTH, OPERATORS
from alphagen.data.expression import BinaryOperator, PairRollingOperator, RollingOperator, UnaryOperator
from alphagen.data.tree import ExpressionBuilder
from alphagen.rl.env.wrapper import (
    OFFSET_CONSTANT,
    OFFSET_DELTA_TIME,
    OFFSET_FEATURE,
    OFFSET_OP,
    OFFSET_SEP,
    SIZE_ACTION,
    SIZE_CONSTANT,
    SIZE_DELTA_TIME,
    SIZE_FEATURE,
    SIZE_OP,
    action2token,
)
from new.classic_factors import ClassicFactor, DEFAULT_LOSS_WEIGHTS
from new.env import HEAD_NAMES, HEAD_TO_ID


SEP_ACTION_ID = OFFSET_SEP - 1
TREND_OPS = {"Ref", "Delta", "Mean", "WMA", "EMA", "TSRank"}
VOLATILITY_OPS = {"Std", "Var", "Mad", "SafeSqrt", "Abs", "Max", "Min"}
VOLUME_FIELDS = {"$volume", "$vwap"}
CORR_OPS = {"Corr", "Cov"}
RANK_OPS = {"CSRank", "TSRank", "Sign", "Greater", "Less", "Max", "Min"}
RISKY_OPS = {"Div", "Log", "Corr", "Cov", "Std", "Var", "Mad"}
TOKEN_OP_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")


def parse_loss_weights(text: str = DEFAULT_LOSS_WEIGHTS) -> Dict[str, float]:
    weights = {"next": 1.0, "head": 0.2, "attr": 0.2}
    if not text:
        return weights
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"invalid pretrain loss weight chunk: {chunk}")
        key, value = chunk.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise ValueError(f"unsupported pretrain loss key: {key}")
        weights[key] = float(value)
    return weights


def _weighted_mean(loss: th.Tensor, weights: th.Tensor) -> th.Tensor:
    if loss.numel() == 0:
        return loss.mean()
    w = weights.to(dtype=loss.dtype, device=loss.device)
    return (loss * w).sum() / w.sum().clamp_min(1e-8)


def _mask_for_prefix(prefix_actions: Sequence[int]) -> np.ndarray:
    builder = ExpressionBuilder()
    for action in prefix_actions:
        builder.add_token(action2token(int(action)))

    valid_op_unary = builder.validate_op(UnaryOperator)
    valid_op_binary = builder.validate_op(BinaryOperator)
    valid_op_rolling = builder.validate_op(RollingOperator)
    valid_op_pair_rolling = builder.validate_op(PairRollingOperator)
    valid = {
        "select": [
            valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling,
            builder.validate_feature(),
            builder.validate_const(),
            builder.validate_dt(),
            builder.is_valid(),
        ],
        "op": {
            UnaryOperator: valid_op_unary,
            BinaryOperator: valid_op_binary,
            RollingOperator: valid_op_rolling,
            PairRollingOperator: valid_op_pair_rolling,
        },
    }

    mask = np.zeros(SIZE_ACTION, dtype=bool)
    for i in range(OFFSET_OP, OFFSET_OP + SIZE_OP):
        if valid["op"][OPERATORS[i - OFFSET_OP].category_type()]:
            mask[i - 1] = True
    if valid["select"][1]:
        for i in range(OFFSET_FEATURE, OFFSET_FEATURE + SIZE_FEATURE):
            mask[i - 1] = True
    if valid["select"][2]:
        for i in range(OFFSET_CONSTANT, OFFSET_CONSTANT + SIZE_CONSTANT):
            mask[i - 1] = True
    if valid["select"][3]:
        for i in range(OFFSET_DELTA_TIME, OFFSET_DELTA_TIME + SIZE_DELTA_TIME):
            mask[i - 1] = True
    if valid["select"][4]:
        mask[SEP_ACTION_ID] = True
    return mask


def _obs_from_actions(actions: Sequence[int], head_id: int) -> np.ndarray:
    obs = np.zeros(MAX_EXPR_LENGTH + 1, dtype=np.uint8)
    if actions:
        obs[: len(actions)] = np.asarray(actions, dtype=np.uint8)
    obs[-1] = head_id
    return obs


def _max_depth_from_actions(actions: Sequence[int]) -> int:
    builder = ExpressionBuilder()
    for action in actions:
        builder.add_token(action2token(int(action)))
    expr = str(builder.get_tree())
    depth = 0
    best = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            best = max(best, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return best


def _token_ops(token_text: Sequence[str]) -> List[str]:
    return [x for x in token_text if TOKEN_OP_RE.match(x) and x in {op.__name__ for op in OPERATORS}]


def _factor_attrs(factor: ClassicFactor) -> Dict[str, int]:
    ops = set(_token_ops(factor.token_text))
    fields = {x for x in factor.token_text if x.startswith("$")}
    length = len(factor.action_ids)
    depth = _max_depth_from_actions(factor.action_ids)
    has_range = "$high" in fields and "$low" in fields
    return {
        "length_bucket": 0 if length <= 5 else (1 if length <= 10 else 2),
        "depth_bucket": 0 if depth <= 3 else (1 if depth <= 6 else 2),
        "has_trend": int(bool(ops & TREND_OPS)),
        "has_volatility": int(bool(ops & VOLATILITY_OPS) or has_range),
        "has_volume": int(bool(fields & VOLUME_FIELDS)),
        "has_corr": int(bool(ops & CORR_OPS)),
        "has_rank": int(bool(ops & RANK_OPS)),
        "has_risky": int(bool(ops & RISKY_OPS)),
    }


def _balanced_factor_weights(factors: Sequence[ClassicFactor]) -> np.ndarray:
    if not factors:
        return np.asarray([], dtype=np.float32)
    head_counts = Counter(f.head for f in factors)
    family_counts = Counter(f.family or f.head for f in factors)
    n = float(len(factors))
    raw = []
    for factor in factors:
        head_balance = n / (max(1, len(head_counts)) * head_counts[factor.head])
        family_key = factor.family or factor.head
        family_balance = n / (max(1, len(family_counts)) * family_counts[family_key])
        raw.append(float(factor.weight) * (head_balance * family_balance) ** 0.5)
    arr = np.asarray(raw, dtype=np.float32)
    return arr / max(1e-8, float(arr.mean()))


def build_pretrain_samples(factors: Sequence[ClassicFactor]) -> Dict[str, np.ndarray]:
    observations: List[np.ndarray] = []
    targets: List[int] = []
    masks: List[np.ndarray] = []
    sample_heads: List[str] = []
    sample_families: List[str] = []
    sample_weights: List[float] = []

    full_observations: List[np.ndarray] = []
    full_heads: List[int] = []
    full_families: List[str] = []
    full_weights: List[float] = []
    length_buckets: List[int] = []
    depth_buckets: List[int] = []
    has_trend: List[int] = []
    has_volatility: List[int] = []
    has_volume: List[int] = []
    has_corr: List[int] = []
    has_rank: List[int] = []
    has_risky: List[int] = []

    factor_weights = _balanced_factor_weights(factors)
    for factor_idx, factor in enumerate(factors):
        head_id = HEAD_TO_ID[factor.head]
        actions = list(factor.action_ids) + [SEP_ACTION_ID]
        prefix: List[int] = []
        weight = float(factor_weights[factor_idx]) if len(factor_weights) else 1.0
        for target in actions:
            if len(prefix) > MAX_EXPR_LENGTH:
                raise ValueError(f"classic factor exceeds MAX_EXPR_LENGTH: {factor.name}")
            mask = _mask_for_prefix(prefix)
            if not mask[int(target)]:
                raise ValueError(f"target action is not legal for factor={factor.name}, target={target}")
            observations.append(_obs_from_actions(prefix, head_id))
            targets.append(int(target))
            masks.append(mask)
            sample_heads.append(factor.head)
            sample_families.append(factor.family or factor.head)
            sample_weights.append(weight)
            prefix.append(int(target))

        attrs = _factor_attrs(factor)
        full_observations.append(_obs_from_actions(factor.action_ids, head_id))
        full_heads.append(head_id)
        full_families.append(factor.family or factor.head)
        full_weights.append(weight)
        length_buckets.append(attrs["length_bucket"])
        depth_buckets.append(attrs["depth_bucket"])
        has_trend.append(attrs["has_trend"])
        has_volatility.append(attrs["has_volatility"])
        has_volume.append(attrs["has_volume"])
        has_corr.append(attrs["has_corr"])
        has_rank.append(attrs["has_rank"])
        has_risky.append(attrs["has_risky"])

    return {
        "observations": np.asarray(observations, dtype=np.uint8),
        "targets": np.asarray(targets, dtype=np.int64),
        "masks": np.asarray(masks, dtype=bool),
        "sample_heads": np.asarray(sample_heads, dtype=object),
        "sample_families": np.asarray(sample_families, dtype=object),
        "sample_weights": np.asarray(sample_weights, dtype=np.float32),
        "full_observations": np.asarray(full_observations, dtype=np.uint8),
        "full_heads": np.asarray(full_heads, dtype=np.int64),
        "full_families": np.asarray(full_families, dtype=object),
        "full_weights": np.asarray(full_weights, dtype=np.float32),
        "length_buckets": np.asarray(length_buckets, dtype=np.int64),
        "depth_buckets": np.asarray(depth_buckets, dtype=np.int64),
        "has_trend": np.asarray(has_trend, dtype=np.float32),
        "has_volatility": np.asarray(has_volatility, dtype=np.float32),
        "has_volume": np.asarray(has_volume, dtype=np.float32),
        "has_corr": np.asarray(has_corr, dtype=np.float32),
        "has_rank": np.asarray(has_rank, dtype=np.float32),
        "has_risky": np.asarray(has_risky, dtype=np.float32),
    }


def _pretrain_parameters(policy) -> List[th.nn.Parameter]:  # type: ignore[no-untyped-def]
    params: List[th.nn.Parameter] = []
    params.extend(policy.features_extractor.parameters())
    params.extend(policy.mlp_extractor.parameters())
    if hasattr(policy, "action_adapters"):
        params.extend(policy.action_adapters.parameters())
    params.extend(policy.action_nets.parameters())
    return [p for p in params if p.requires_grad]


def _actor_latent(policy, obs: th.Tensor) -> th.Tensor:  # type: ignore[no-untyped-def]
    features = policy.extract_features(obs)
    if not policy.share_features_extractor:
        features = features[0]
    return policy.mlp_extractor.forward_actor(features)


def _aux_loss(
    model,
    aux_modules: nn.ModuleDict,
    obs_np: np.ndarray,
    weights_np: np.ndarray,
    labels: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> Dict[str, th.Tensor]:
    device = model.device
    obs = th.as_tensor(obs_np[indices], dtype=th.float32, device=device)
    weights = th.as_tensor(weights_np[indices], dtype=th.float32, device=device)
    latent = _actor_latent(model.policy, obs)

    head_targets = th.as_tensor(labels["head"][indices], dtype=th.long, device=device)
    length_targets = th.as_tensor(labels["length_bucket"][indices], dtype=th.long, device=device)
    depth_targets = th.as_tensor(labels["depth_bucket"][indices], dtype=th.long, device=device)
    has_trend = th.as_tensor(labels["has_trend"][indices], dtype=th.float32, device=device)
    has_volatility = th.as_tensor(labels["has_volatility"][indices], dtype=th.float32, device=device)
    has_volume = th.as_tensor(labels["has_volume"][indices], dtype=th.float32, device=device)
    has_corr = th.as_tensor(labels["has_corr"][indices], dtype=th.float32, device=device)
    has_rank = th.as_tensor(labels["has_rank"][indices], dtype=th.float32, device=device)
    has_risky = th.as_tensor(labels["has_risky"][indices], dtype=th.float32, device=device)

    head_loss = _weighted_mean(F.cross_entropy(aux_modules["head"](latent), head_targets, reduction="none"), weights)
    length_loss = _weighted_mean(F.cross_entropy(aux_modules["length"](latent), length_targets, reduction="none"), weights)
    depth_loss = _weighted_mean(F.cross_entropy(aux_modules["depth"](latent), depth_targets, reduction="none"), weights)
    trend_loss = _weighted_mean(F.binary_cross_entropy_with_logits(aux_modules["has_trend"](latent).squeeze(-1), has_trend, reduction="none"), weights)
    volatility_loss = _weighted_mean(F.binary_cross_entropy_with_logits(aux_modules["has_volatility"](latent).squeeze(-1), has_volatility, reduction="none"), weights)
    volume_loss = _weighted_mean(F.binary_cross_entropy_with_logits(aux_modules["has_volume"](latent).squeeze(-1), has_volume, reduction="none"), weights)
    corr_loss = _weighted_mean(F.binary_cross_entropy_with_logits(aux_modules["has_corr"](latent).squeeze(-1), has_corr, reduction="none"), weights)
    rank_loss = _weighted_mean(F.binary_cross_entropy_with_logits(aux_modules["has_rank"](latent).squeeze(-1), has_rank, reduction="none"), weights)
    risky_loss = _weighted_mean(F.binary_cross_entropy_with_logits(aux_modules["has_risky"](latent).squeeze(-1), has_risky, reduction="none"), weights)
    attr_loss = (
        length_loss
        + depth_loss
        + trend_loss
        + volatility_loss
        + volume_loss
        + corr_loss
        + rank_loss
        + risky_loss
    ) / 8.0
    return {"head": head_loss, "attr": attr_loss}


def pretrain_policy_heads(
    model,
    factors: Sequence[ClassicFactor],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = 0,
    loss_weights: Optional[Mapping[str, float]] = None,
    use_aux_loss: bool = True,
) -> Dict:
    parsed_loss_weights = {"next": 1.0, "head": 0.2, "attr": 0.2}
    if loss_weights is not None:
        parsed_loss_weights.update({k: float(v) for k, v in loss_weights.items()})
    if not use_aux_loss:
        parsed_loss_weights["head"] = 0.0
        parsed_loss_weights["attr"] = 0.0

    if epochs <= 0 or not factors:
        return {
            "enabled": False,
            "epochs": int(epochs),
            "factor_count": len(factors),
            "sample_count": 0,
            "loss_start": None,
            "loss_end": None,
            "losses": [],
            "next_losses": [],
            "head_losses": [],
            "attr_losses": [],
            "aux_loss_enabled": bool(use_aux_loss),
            "loss_weights": parsed_loss_weights,
        }

    dataset = build_pretrain_samples(factors)
    obs_np = dataset["observations"]
    target_np = dataset["targets"]
    mask_np = dataset["masks"]
    sample_heads = dataset["sample_heads"]
    sample_families = dataset["sample_families"]
    sample_weights_np = dataset["sample_weights"]

    full_obs_np = dataset["full_observations"]
    full_weights_np = dataset["full_weights"]
    full_labels = {
        "head": dataset["full_heads"],
        "length_bucket": dataset["length_buckets"],
        "depth_bucket": dataset["depth_buckets"],
        "has_trend": dataset["has_trend"],
        "has_volatility": dataset["has_volatility"],
        "has_volume": dataset["has_volume"],
        "has_corr": dataset["has_corr"],
        "has_rank": dataset["has_rank"],
        "has_risky": dataset["has_risky"],
    }

    device = model.device
    latent_dim = int(model.policy.mlp_extractor.latent_dim_pi)
    aux_modules = nn.ModuleDict(
        {
            "head": nn.Linear(latent_dim, len(HEAD_NAMES)),
            "length": nn.Linear(latent_dim, 3),
            "depth": nn.Linear(latent_dim, 3),
            "has_trend": nn.Linear(latent_dim, 1),
            "has_volatility": nn.Linear(latent_dim, 1),
            "has_volume": nn.Linear(latent_dim, 1),
            "has_corr": nn.Linear(latent_dim, 1),
            "has_rank": nn.Linear(latent_dim, 1),
            "has_risky": nn.Linear(latent_dim, 1),
        }
    ).to(device)
    params = _pretrain_parameters(model.policy)
    if use_aux_loss:
        params.extend(aux_modules.parameters())
    optimizer = th.optim.Adam(params, lr=float(lr))
    rng = np.random.default_rng(seed)
    losses: List[float] = []
    next_losses: List[float] = []
    head_losses: List[float] = []
    attr_losses: List[float] = []
    n = len(target_np)
    n_full = len(full_obs_np)
    model.policy.train()
    aux_modules.train()

    for epoch in range(int(epochs)):
        indices = rng.permutation(n)
        full_indices = rng.permutation(n_full)
        epoch_loss = 0.0
        epoch_next = 0.0
        epoch_head = 0.0
        epoch_attr = 0.0
        seen = 0
        for batch_no, start in enumerate(range(0, n, int(batch_size))):
            batch_idx = indices[start : start + int(batch_size)]
            obs = th.as_tensor(obs_np[batch_idx], dtype=th.float32, device=device)
            targets = th.as_tensor(target_np[batch_idx], dtype=th.long, device=device)
            masks = th.as_tensor(mask_np[batch_idx], dtype=th.bool, device=device)
            weights = th.as_tensor(sample_weights_np[batch_idx], dtype=th.float32, device=device)
            logits = model.policy.get_action_logits(obs).masked_fill(~masks, -1e9)
            next_loss = _weighted_mean(F.cross_entropy(logits, targets, reduction="none"), weights)

            head_loss = th.zeros((), dtype=next_loss.dtype, device=device)
            attr_loss = th.zeros((), dtype=next_loss.dtype, device=device)
            if use_aux_loss and (parsed_loss_weights["head"] > 0.0 or parsed_loss_weights["attr"] > 0.0):
                aux_start = (batch_no * int(batch_size)) % n_full
                aux_idx = full_indices[aux_start : aux_start + int(batch_size)]
                if len(aux_idx) < int(batch_size):
                    aux_idx = np.concatenate([aux_idx, full_indices[: int(batch_size) - len(aux_idx)]])
                aux = _aux_loss(
                    model=model,
                    aux_modules=aux_modules,
                    obs_np=full_obs_np,
                    weights_np=full_weights_np,
                    labels=full_labels,
                    indices=aux_idx,
                )
                head_loss = aux["head"]
                attr_loss = aux["attr"]

            loss = (
                parsed_loss_weights["next"] * next_loss
                + parsed_loss_weights["head"] * head_loss
                + parsed_loss_weights["attr"] * attr_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size_actual = len(batch_idx)
            epoch_loss += float(loss.detach().cpu()) * batch_size_actual
            epoch_next += float(next_loss.detach().cpu()) * batch_size_actual
            epoch_head += float(head_loss.detach().cpu()) * batch_size_actual
            epoch_attr += float(attr_loss.detach().cpu()) * batch_size_actual
            seen += batch_size_actual
        losses.append(epoch_loss / max(1, seen))
        next_losses.append(epoch_next / max(1, seen))
        head_losses.append(epoch_head / max(1, seen))
        attr_losses.append(epoch_attr / max(1, seen))
        print(
            f"[head-pretrain] epoch={epoch + 1}/{epochs} "
            f"loss={losses[-1]:.6f} next={next_losses[-1]:.6f} "
            f"head={head_losses[-1]:.6f} attr={attr_losses[-1]:.6f}"
        )

    return {
        "enabled": True,
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "factor_count": len(factors),
        "sample_count": int(n),
        "full_expr_sample_count": int(n_full),
        "factor_head_counts": dict(Counter(f.head for f in factors)),
        "factor_family_counts": dict(Counter((f.family or f.head) for f in factors)),
        "factor_source_counts": dict(Counter(f.source for f in factors)),
        "sample_head_counts": dict(Counter(str(x) for x in sample_heads)),
        "sample_family_counts": dict(Counter(str(x) for x in sample_families)),
        "loss_start": float(losses[0]) if losses else None,
        "loss_end": float(losses[-1]) if losses else None,
        "next_loss_start": float(next_losses[0]) if next_losses else None,
        "next_loss_end": float(next_losses[-1]) if next_losses else None,
        "head_loss_start": float(head_losses[0]) if head_losses else None,
        "head_loss_end": float(head_losses[-1]) if head_losses else None,
        "attr_loss_start": float(attr_losses[0]) if attr_losses else None,
        "attr_loss_end": float(attr_losses[-1]) if attr_losses else None,
        "losses": [float(x) for x in losses],
        "next_losses": [float(x) for x in next_losses],
        "head_losses": [float(x) for x in head_losses],
        "attr_losses": [float(x) for x in attr_losses],
        "aux_loss_enabled": bool(use_aux_loss),
        "loss_weights": parsed_loss_weights,
        "factor_weight_mean": float(np.mean(dataset["full_weights"])) if len(dataset["full_weights"]) else 0.0,
        "factor_weight_min": float(np.min(dataset["full_weights"])) if len(dataset["full_weights"]) else 0.0,
        "factor_weight_max": float(np.max(dataset["full_weights"])) if len(dataset["full_weights"]) else 0.0,
    }
