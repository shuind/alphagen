from collections import Counter
from typing import Dict, List, Optional, Tuple

from alphagen.data.expression import Expression
from alphagen.models.alpha_pool import AlphaPool

from new.intrinsic import AstCountIntrinsic, ast_signature


class MultiHeadAlphaPool(AlphaPool):
    """Thin clean wrapper around AlphaPool for source-head tracking and AST intrinsic reward."""

    def __init__(self, *args, intrinsic_beta: float = 0.0, **kwargs) -> None:
        super().__init__(*args, reward_mode="re", lambda_ri=0.0, **kwargs)
        self.intrinsic_beta = float(intrinsic_beta)
        self.intrinsic = AstCountIntrinsic()
        self.expr_source_heads: List[Optional[str]] = [None for _ in range(self.capacity + 1)]
        self.head_generated = Counter()
        self.head_valid = Counter()
        self.head_accepted = Counter()
        self._pending_source_head = "base"

    def try_new_expr(
        self,
        expr: Expression,
        token_seq: Optional[List[str]] = None,
        source_head: str = "base",
    ) -> Tuple[float, Dict]:
        self._pending_source_head = source_head
        self.head_generated[source_head] += 1
        reward, info = super().try_new_expr(expr, token_seq=token_seq)
        info = dict(info)
        info["source_head"] = source_head
        info["source_strategy"] = source_head

        if info.get("invalid", False):
            self.last_reward_info = info
            return reward, info

        self.head_valid[source_head] += 1
        signature = ast_signature(expr)
        intrinsic_raw = self.intrinsic.reward(signature)
        self.intrinsic.update(signature)
        intrinsic_reward = self.intrinsic_beta * intrinsic_raw if source_head == "explore" else 0.0
        reward = float(reward + intrinsic_reward)

        expr_text = str(expr)
        survived = any(str(e) == expr_text for e in self.exprs[: self.size] if e is not None)
        if survived:
            self.head_accepted[source_head] += 1

        info.update(
            {
                "ast_signature": signature,
                "ast_count_before": int(self.intrinsic.counts[signature] - 1),
                "intrinsic_raw": float(intrinsic_raw),
                "intrinsic_beta": float(self.intrinsic_beta),
                "intrinsic_reward": float(intrinsic_reward),
                "reward_total": float(reward),
                "head_generated": dict(self.head_generated),
                "head_valid": dict(self.head_valid),
                "head_accepted": dict(self.head_accepted),
                "strategy_generated": dict(self.head_generated),
                "strategy_valid": dict(self.head_valid),
                "strategy_accepted": dict(self.head_accepted),
            }
        )
        self.last_reward_info = info
        return reward, info

    def _add_factor(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super()._add_factor(*args, **kwargs)
        self.expr_source_heads[self.size - 1] = self._pending_source_head

    def _swap_idx(self, i, j) -> None:
        super()._swap_idx(i, j)
        self.expr_source_heads[i], self.expr_source_heads[j] = self.expr_source_heads[j], self.expr_source_heads[i]

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload["source_heads"] = list(self.expr_source_heads[: self.size])
        payload["source_strategies"] = list(self.expr_source_heads[: self.size])
        payload["head_generated"] = dict(self.head_generated)
        payload["head_valid"] = dict(self.head_valid)
        payload["head_accepted"] = dict(self.head_accepted)
        payload["strategy_generated"] = dict(self.head_generated)
        payload["strategy_valid"] = dict(self.head_valid)
        payload["strategy_accepted"] = dict(self.head_accepted)
        payload["ast_counts"] = self.intrinsic.to_dict()
        payload["intrinsic_beta"] = self.intrinsic_beta
        return payload
