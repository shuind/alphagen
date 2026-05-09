from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import math

from alphagen.data.expression import Expression
from alphagen.models.alpha_pool import AlphaPool


class MotifEditAlphaPool(AlphaPool):
    """AlphaPool wrapper for motif-edit generation metadata and behavior novelty."""

    def __init__(
        self,
        *args,
        behavior_novelty_beta: float = 0.0,
        motif_prior_eta: float = 0.02,
        behavior_archive_size: int = 128,
        behavior_novelty_strategies: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, reward_mode="re", lambda_ri=0.0, **kwargs)
        self.behavior_novelty_beta = float(behavior_novelty_beta)
        self.motif_prior_eta = float(motif_prior_eta)
        self.behavior_archive_size = max(0, int(behavior_archive_size))
        self.behavior_novelty_strategies = None if behavior_novelty_strategies is None else set(behavior_novelty_strategies)
        self.behavior_archive: Deque[Expression] = deque(maxlen=self.behavior_archive_size)

        self.expr_source_heads: List[Optional[str]] = [None for _ in range(self.capacity + 1)]
        self.expr_motif_ids: List[Optional[str]] = [None for _ in range(self.capacity + 1)]
        self.expr_motif_families: List[Optional[str]] = [None for _ in range(self.capacity + 1)]
        self.expr_edit_paths: List[Optional[List[str]]] = [None for _ in range(self.capacity + 1)]
        self.expr_naturalness_scores: List[Optional[float]] = [None for _ in range(self.capacity + 1)]
        self.expr_behavior_novelty_scores: List[Optional[float]] = [None for _ in range(self.capacity + 1)]

        self.head_generated = Counter()
        self.head_valid = Counter()
        self.head_accepted = Counter()
        self.motif_generated = Counter()
        self.motif_accepted = Counter()
        self.family_generated = Counter()
        self.family_accepted = Counter()

        self._pending_source_head = "base"
        self._pending_motif_id = ""
        self._pending_motif_family = ""
        self._pending_edit_path: List[str] = []
        self._pending_naturalness_score = 0.0
        self._pending_behavior_novelty = 0.0

    def _calc_behavior_novelty(self, expr: Expression) -> float:
        if not self.behavior_archive:
            return 1.0
        max_abs = 0.0
        for archived in self.behavior_archive:
            try:
                max_abs = max(max_abs, abs(float(self._get_mutual_ic_cached(expr, archived))))
            except Exception:
                continue
        return float(max(0.0, 1.0 - max_abs))

    def _novelty_enabled_for_strategy(self, source_head: str) -> bool:
        if self.behavior_novelty_beta <= 0.0 or self.behavior_archive_size <= 0:
            return False
        return self.behavior_novelty_strategies is None or source_head in self.behavior_novelty_strategies

    def try_new_expr(
        self,
        expr: Expression,
        token_seq: Optional[List[str]] = None,
        source_head: str = "base",
        motif_id: str = "",
        motif_family: str = "",
        edit_path: Optional[List[str]] = None,
        naturalness_score: float = 1.0,
        naturalness_passed: bool = True,
        naturalness_reasons: Optional[List[str]] = None,
        naturalness_stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict]:
        edit_path = list(edit_path or [])
        naturalness_reasons = list(naturalness_reasons or [])
        naturalness_stats = dict(naturalness_stats or {})
        self._pending_source_head = source_head
        self._pending_motif_id = motif_id
        self._pending_motif_family = motif_family
        self._pending_edit_path = edit_path
        self._pending_naturalness_score = float(naturalness_score)
        self.head_generated[source_head] += 1
        self.motif_generated[motif_id] += 1
        self.family_generated[motif_family] += 1

        if not naturalness_passed:
            info = {
                "re": 0.0,
                "reward_total": -1.0,
                "reward_pool": 0.0,
                "invalid": True,
                "invalid_reason": "naturalness_failed",
                "source_head": source_head,
                "source_strategy": source_head,
                "motif_id": motif_id,
                "motif_family": motif_family,
                "edit_path": edit_path,
                "naturalness_score": float(naturalness_score),
                "naturalness_passed": False,
                "naturalness_reasons": naturalness_reasons,
                "naturalness_stats": naturalness_stats,
                "behavior_novelty": 0.0,
                "behavior_novelty_reward": 0.0,
                "naturalness_reward": 0.0,
            }
            self.last_reward_info = info
            return -1.0, info

        novelty_enabled = self._novelty_enabled_for_strategy(source_head)
        behavior_novelty = self._calc_behavior_novelty(expr) if novelty_enabled else 0.0
        self._pending_behavior_novelty = behavior_novelty
        reward, info = super().try_new_expr(expr, token_seq=token_seq)
        info = dict(info)
        info.update(
            {
                "source_head": source_head,
                "source_strategy": source_head,
                "motif_id": motif_id,
                "motif_family": motif_family,
                "edit_path": edit_path,
                "naturalness_score": float(naturalness_score),
                "naturalness_passed": True,
                "naturalness_reasons": naturalness_reasons,
                "naturalness_stats": naturalness_stats,
                "behavior_novelty": float(behavior_novelty),
                "behavior_novelty_beta": float(self.behavior_novelty_beta),
                "motif_prior_eta": float(self.motif_prior_eta),
            }
        )

        if info.get("invalid", False):
            self.last_reward_info = info
            return reward, info

        self.head_valid[source_head] += 1
        behavior_reward = (self.behavior_novelty_beta * behavior_novelty) if novelty_enabled else 0.0
        has_motif_trace = bool(motif_id or edit_path)
        naturalness_reward = (self.motif_prior_eta * float(naturalness_score)) if has_motif_trace else 0.0
        reward_total = float(reward + behavior_reward + naturalness_reward)

        re_value = float(info.get("re", 0.0))
        if novelty_enabled and re_value > 0:
            self.behavior_archive.append(expr)

        expr_text = str(expr)
        survived = any(str(e) == expr_text for e in self.exprs[: self.size] if e is not None)
        if survived:
            self.head_accepted[source_head] += 1
            self.motif_accepted[motif_id] += 1
            self.family_accepted[motif_family] += 1

        info.update(
            {
                "behavior_novelty_reward": float(behavior_reward),
                "naturalness_reward": float(naturalness_reward),
                "reward_total": reward_total,
                "reward_pool": float(info.get("reward_pool", reward)),
                "behavior_archive_size": int(len(self.behavior_archive)),
                "head_generated": dict(self.head_generated),
                "head_valid": dict(self.head_valid),
                "head_accepted": dict(self.head_accepted),
                "strategy_generated": dict(self.head_generated),
                "strategy_valid": dict(self.head_valid),
                "strategy_accepted": dict(self.head_accepted),
                "motif_generated": dict(self.motif_generated),
                "motif_accepted": dict(self.motif_accepted),
                "motif_family_generated": dict(self.family_generated),
                "motif_family_accepted": dict(self.family_accepted),
            }
        )
        self.last_reward_info = info
        return reward_total, info

    def _add_factor(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super()._add_factor(*args, **kwargs)
        idx = self.size - 1
        self.expr_source_heads[idx] = self._pending_source_head
        self.expr_motif_ids[idx] = self._pending_motif_id
        self.expr_motif_families[idx] = self._pending_motif_family
        self.expr_edit_paths[idx] = list(self._pending_edit_path)
        self.expr_naturalness_scores[idx] = float(self._pending_naturalness_score)
        self.expr_behavior_novelty_scores[idx] = float(self._pending_behavior_novelty)

    def _swap_idx(self, i, j) -> None:
        super()._swap_idx(i, j)
        self.expr_source_heads[i], self.expr_source_heads[j] = self.expr_source_heads[j], self.expr_source_heads[i]
        self.expr_motif_ids[i], self.expr_motif_ids[j] = self.expr_motif_ids[j], self.expr_motif_ids[i]
        self.expr_motif_families[i], self.expr_motif_families[j] = self.expr_motif_families[j], self.expr_motif_families[i]
        self.expr_edit_paths[i], self.expr_edit_paths[j] = self.expr_edit_paths[j], self.expr_edit_paths[i]
        self.expr_naturalness_scores[i], self.expr_naturalness_scores[j] = (
            self.expr_naturalness_scores[j],
            self.expr_naturalness_scores[i],
        )
        self.expr_behavior_novelty_scores[i], self.expr_behavior_novelty_scores[j] = (
            self.expr_behavior_novelty_scores[j],
            self.expr_behavior_novelty_scores[i],
        )

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload["source_heads"] = list(self.expr_source_heads[: self.size])
        payload["source_strategies"] = list(self.expr_source_heads[: self.size])
        payload["motif_ids"] = list(self.expr_motif_ids[: self.size])
        payload["motif_families"] = list(self.expr_motif_families[: self.size])
        payload["edit_paths"] = list(self.expr_edit_paths[: self.size])
        payload["naturalness_scores"] = [
            (None if value is None or not math.isfinite(float(value)) else float(value))
            for value in self.expr_naturalness_scores[: self.size]
        ]
        payload["behavior_novelty_scores"] = [
            (None if value is None or not math.isfinite(float(value)) else float(value))
            for value in self.expr_behavior_novelty_scores[: self.size]
        ]
        payload["head_generated"] = dict(self.head_generated)
        payload["head_valid"] = dict(self.head_valid)
        payload["head_accepted"] = dict(self.head_accepted)
        payload["strategy_generated"] = dict(self.head_generated)
        payload["strategy_valid"] = dict(self.head_valid)
        payload["strategy_accepted"] = dict(self.head_accepted)
        payload["motif_generated"] = dict(self.motif_generated)
        payload["motif_accepted"] = dict(self.motif_accepted)
        payload["motif_family_generated"] = dict(self.family_generated)
        payload["motif_family_accepted"] = dict(self.family_accepted)
        payload["behavior_novelty_beta"] = self.behavior_novelty_beta
        payload["behavior_novelty_strategies"] = (
            None if self.behavior_novelty_strategies is None else sorted(self.behavior_novelty_strategies)
        )
        payload["motif_prior_eta"] = self.motif_prior_eta
        payload["behavior_archive_size"] = len(self.behavior_archive)
        payload["behavior_archive_max_size"] = self.behavior_archive_size
        return payload
