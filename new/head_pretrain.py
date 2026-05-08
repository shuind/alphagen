from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
import torch as th
import torch.nn.functional as F

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
from new.classic_factors import ClassicFactor
from new.env import HEAD_TO_ID


SEP_ACTION_ID = OFFSET_SEP - 1


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


def build_pretrain_samples(factors: Sequence[ClassicFactor]) -> Dict[str, np.ndarray]:
    observations: List[np.ndarray] = []
    targets: List[int] = []
    masks: List[np.ndarray] = []
    sample_heads: List[str] = []

    for factor in factors:
        head_id = HEAD_TO_ID[factor.head]
        actions = list(factor.action_ids) + [SEP_ACTION_ID]
        prefix: List[int] = []
        for target in actions:
            if len(prefix) > MAX_EXPR_LENGTH:
                raise ValueError(f"classic factor exceeds MAX_EXPR_LENGTH: {factor.name}")
            obs = np.zeros(MAX_EXPR_LENGTH + 1, dtype=np.uint8)
            if prefix:
                obs[: len(prefix)] = np.asarray(prefix, dtype=np.uint8)
            obs[-1] = head_id
            mask = _mask_for_prefix(prefix)
            if not mask[int(target)]:
                raise ValueError(f"target action is not legal for factor={factor.name}, target={target}")
            observations.append(obs)
            targets.append(int(target))
            masks.append(mask)
            sample_heads.append(factor.head)
            prefix.append(int(target))

    return {
        "observations": np.asarray(observations, dtype=np.uint8),
        "targets": np.asarray(targets, dtype=np.int64),
        "masks": np.asarray(masks, dtype=bool),
        "sample_heads": np.asarray(sample_heads, dtype=object),
    }


def _pretrain_parameters(policy) -> List[th.nn.Parameter]:  # type: ignore[no-untyped-def]
    params: List[th.nn.Parameter] = []
    params.extend(policy.features_extractor.parameters())
    params.extend(policy.mlp_extractor.parameters())
    params.extend(policy.action_nets.parameters())
    return [p for p in params if p.requires_grad]


def pretrain_policy_heads(
    model,
    factors: Sequence[ClassicFactor],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = 0,
) -> Dict:
    if epochs <= 0 or not factors:
        return {
            "enabled": False,
            "epochs": int(epochs),
            "factor_count": len(factors),
            "sample_count": 0,
            "loss_start": None,
            "loss_end": None,
            "losses": [],
        }

    dataset = build_pretrain_samples(factors)
    obs_np = dataset["observations"]
    target_np = dataset["targets"]
    mask_np = dataset["masks"]
    sample_heads = dataset["sample_heads"]

    device = model.device
    optimizer = th.optim.Adam(_pretrain_parameters(model.policy), lr=float(lr))
    rng = np.random.default_rng(seed)
    losses: List[float] = []
    n = len(target_np)
    model.policy.train()

    for epoch in range(int(epochs)):
        indices = rng.permutation(n)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, n, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            obs = th.as_tensor(obs_np[batch_idx], dtype=th.float32, device=device)
            targets = th.as_tensor(target_np[batch_idx], dtype=th.long, device=device)
            masks = th.as_tensor(mask_np[batch_idx], dtype=th.bool, device=device)
            logits = model.policy.get_action_logits(obs)
            logits = logits.masked_fill(~masks, -1e9)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size_actual = len(batch_idx)
            epoch_loss += float(loss.detach().cpu()) * batch_size_actual
            seen += batch_size_actual
        losses.append(epoch_loss / max(1, seen))
        print(f"[head-pretrain] epoch={epoch + 1}/{epochs} loss={losses[-1]:.6f}")

    return {
        "enabled": True,
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "factor_count": len(factors),
        "sample_count": int(n),
        "factor_head_counts": dict(Counter(f.head for f in factors)),
        "sample_head_counts": dict(Counter(str(x) for x in sample_heads)),
        "loss_start": float(losses[0]) if losses else None,
        "loss_end": float(losses[-1]) if losses else None,
        "losses": [float(x) for x in losses],
    }
