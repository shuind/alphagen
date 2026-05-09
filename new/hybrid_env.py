from __future__ import annotations

from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from alphagen.config import MAX_EXPR_LENGTH, OPERATORS, REWARD_PER_STEP
from alphagen.data.expression import OutOfDataRangeError
from alphagen.rl.env.wrapper import (
    OFFSET_CONSTANT,
    OFFSET_DELTA_TIME,
    OFFSET_FEATURE,
    OFFSET_OP,
    OFFSET_SEP,
    SIZE_ACTION,
    SIZE_ALL,
    SIZE_CONSTANT,
    SIZE_DELTA_TIME,
    SIZE_FEATURE,
    SIZE_OP,
    action2token,
)
from new.env import MultiHeadAlphaEnvCore, STRATEGY_NAMES, STRATEGY_TO_ID
from new.motif_bank import (
    FIELD_NAMES,
    MOTIF_BANK,
    MOTIFS_BY_HEAD,
    PRICE_FIELDS,
    SMOOTH_OPS,
    VOLUME_FIELDS,
    WINDOWS,
    build_motif_expression,
    default_state,
    naturalness_score,
)
from new.motif_env import EDIT_ACTIONS, N_ACTIONS as MOTIF_ACTIONS, N_MOTIF_ACTIONS, OBS_DIM as MOTIF_OBS_DIM, STOP_ACTION


FREE_TOKEN_STRATEGIES = {"base", "explore"}
MOTIF_STRATEGIES = tuple(name for name in STRATEGY_NAMES if name not in FREE_TOKEN_STRATEGIES)
HYBRID_ACTIONS = max(SIZE_ACTION, MOTIF_ACTIONS)
HYBRID_OBS_DIM = 2 + max(MAX_EXPR_LENGTH + 1, MOTIF_OBS_DIM)


class HybridStrategyAlphaEnv(gym.Env):
    """Hybrid search environment.

    `base` and `explore` keep the original free-token AlphaGen DSL. The finance
    family strategies use motif selection plus bounded semantic edit actions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pool,
        method: str = "hybrid_strategy_intrinsic",
        max_edits: int = 4,
        reward_per_step: float = REWARD_PER_STEP,
        **token_core_kwargs,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.method = method
        self.max_edits = max(0, int(max_edits))
        self.reward_per_step = float(reward_per_step)
        self.action_space = gym.spaces.Discrete(HYBRID_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(HYBRID_OBS_DIM,), dtype=np.float32)

        self.token_core = MultiHeadAlphaEnvCore(pool=pool, **token_core_kwargs)
        self._episode_idx = -1
        self._strategy_id = 0
        self.current_strategy = "base"
        self._mode = "token"

        self._token_state = np.zeros(MAX_EXPR_LENGTH + 1, dtype=np.uint8)
        self._token_counter = 0

        self._motif_state: Dict[str, object] | None = None
        self._motif_edit_count = 0
        self._motif_edit_path: List[str] = []

    @property
    def unwrapped(self):  # type: ignore[override]
        return self

    def _choose_strategy(self) -> int:
        return self._episode_idx % len(STRATEGY_NAMES)

    def _is_token_mode(self) -> bool:
        return self.current_strategy in FREE_TOKEN_STRATEGIES

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._episode_idx += 1
        self._strategy_id = self._choose_strategy()
        self.current_strategy = STRATEGY_NAMES[self._strategy_id]
        self._mode = "token" if self._is_token_mode() else "motif"

        self._token_state = np.zeros(MAX_EXPR_LENGTH + 1, dtype=np.uint8)
        self._token_state[-1] = self._strategy_id
        self._token_counter = 0
        self.token_core.current_head = self.current_strategy
        self.token_core.reset()

        self._motif_state = None
        self._motif_edit_count = 0
        self._motif_edit_path = []

        return self._obs(), {
            "source_head": self.current_strategy,
            "source_strategy": self.current_strategy,
            "head_id": self._strategy_id,
            "strategy_id": self._strategy_id,
            "strategy_mode": self._mode,
        }

    def _obs(self) -> np.ndarray:
        obs = np.zeros(HYBRID_OBS_DIM, dtype=np.float32)
        obs[0] = self._strategy_id / max(1, len(STRATEGY_NAMES) - 1)
        obs[1] = 0.0 if self._mode == "token" else 1.0
        if self._mode == "token":
            payload = self._token_state.astype(np.float32) / max(1, SIZE_ALL - 1)
            obs[2 : 2 + len(payload)] = payload
            return obs

        payload = self._motif_obs()
        obs[2 : 2 + len(payload)] = payload
        return obs

    def _motif_obs(self) -> np.ndarray:
        obs = np.zeros(MOTIF_OBS_DIM, dtype=np.float32)
        obs[0] = self._strategy_id / max(1, len(STRATEGY_NAMES) - 1)
        obs[1] = 1.0 if self._motif_state is not None else 0.0
        if self._motif_state is None:
            return obs
        motif_idx = int(self._motif_state.get("motif_index", 0))
        obs[2] = motif_idx / max(1, len(MOTIF_BANK) - 1)
        obs[3] = self._motif_edit_count / max(1, self.max_edits)
        obs[4] = FIELD_NAMES.index(str(self._motif_state.get("field", "close"))) / max(1, len(FIELD_NAMES) - 1)
        obs[5] = FIELD_NAMES.index(str(self._motif_state.get("field2", "open"))) / max(1, len(FIELD_NAMES) - 1)
        obs[6] = PRICE_FIELDS.index(str(self._motif_state.get("price_field", "close"))) / max(1, len(PRICE_FIELDS) - 1)
        obs[7] = VOLUME_FIELDS.index(str(self._motif_state.get("volume_field", "volume"))) / max(1, len(VOLUME_FIELDS) - 1)
        obs[8] = WINDOWS.index(int(self._motif_state.get("window", 20))) / max(1, len(WINDOWS) - 1)
        obs[9] = SMOOTH_OPS.index(str(self._motif_state.get("smooth_op", "mean"))) / max(1, len(SMOOTH_OPS) - 1)
        obs[10] = 1.0 if bool(self._motif_state.get("rank_wrapper", False)) else 0.0
        obs[11] = 1.0 if bool(self._motif_state.get("ts_smooth", False)) else 0.0
        obs[12] = 1.0 if bool(self._motif_state.get("vol_adjust", False)) else 0.0
        return obs

    def _token_action_masks(self) -> np.ndarray:
        mask = np.zeros(HYBRID_ACTIONS, dtype=bool)
        valid = self.token_core.valid_action_types()
        for i in range(OFFSET_OP, OFFSET_OP + SIZE_OP):
            if valid["op"][OPERATORS[i - OFFSET_OP].category_type()]:
                mask[i - 1] = True
        if valid["select"][1]:
            mask[OFFSET_FEATURE - 1 : OFFSET_FEATURE - 1 + SIZE_FEATURE] = True
        if valid["select"][2]:
            mask[OFFSET_CONSTANT - 1 : OFFSET_CONSTANT - 1 + SIZE_CONSTANT] = True
        if valid["select"][3]:
            mask[OFFSET_DELTA_TIME - 1 : OFFSET_DELTA_TIME - 1 + SIZE_DELTA_TIME] = True
        if valid["select"][4]:
            mask[OFFSET_SEP - 1] = True
        return mask

    def _current_motif(self):
        if self._motif_state is None:
            return None
        return MOTIF_BANK[int(self._motif_state["motif_index"])]

    def _motif_edit_valid(self, kind: str, value: str) -> bool:
        if self._motif_state is None or self._motif_edit_count >= self.max_edits:
            return False
        motif = self._current_motif()
        if motif is None:
            return False
        if kind == "field":
            return motif.allow_field and value != self._motif_state.get("field")
        if kind == "field2":
            return motif.allow_field2 and value != self._motif_state.get("field2")
        if kind == "price_field":
            return motif.allow_price and value != self._motif_state.get("price_field")
        if kind == "volume_field":
            return motif.allow_volume and value != self._motif_state.get("volume_field")
        if kind == "window":
            return motif.allow_window and int(value) != int(self._motif_state.get("window", 20))
        if kind == "smooth_op":
            return (motif.allow_smooth_op or bool(self._motif_state.get("ts_smooth", False))) and value != self._motif_state.get("smooth_op")
        if kind == "toggle" and value == "rank_wrapper":
            return motif.allow_rank_wrapper and not bool(self._motif_state.get("rank_wrapper", False))
        if kind == "toggle" and value == "ts_smooth":
            return motif.allow_ts_smooth and not bool(self._motif_state.get("ts_smooth", False))
        if kind == "toggle" and value == "vol_adjust":
            return motif.allow_vol_adjust and not bool(self._motif_state.get("vol_adjust", False))
        return False

    def _motif_action_masks(self) -> np.ndarray:
        mask = np.zeros(HYBRID_ACTIONS, dtype=bool)
        if self._motif_state is None:
            for idx in MOTIFS_BY_HEAD.get(self.current_strategy, []):
                mask[idx] = True
            return mask

        if self._motif_edit_count >= self.max_edits:
            mask[STOP_ACTION] = True
            return mask

        for idx in MOTIFS_BY_HEAD.get(self.current_strategy, []):
            if idx != int(self._motif_state.get("motif_index", -1)):
                mask[idx] = True
        for offset, (kind, value) in enumerate(EDIT_ACTIONS):
            action = N_MOTIF_ACTIONS + offset
            if kind == "stop":
                mask[action] = True
            elif self._motif_edit_valid(kind, value):
                mask[action] = True
        return mask

    def action_masks(self) -> np.ndarray:
        return self._token_action_masks() if self._mode == "token" else self._motif_action_masks()

    def _step_token(self, action: int):
        _, reward, done, truncated, info = self.token_core.step(action2token(action))
        if not done:
            self._token_state[self._token_counter] = action
            self._token_counter += 1
        self._token_state[-1] = self._strategy_id
        total_reward = float(reward + self.reward_per_step)
        info = dict(info) if info is not None else {}
        info.setdefault("source_head", self.current_strategy)
        info.setdefault("source_strategy", self.current_strategy)
        info["head_id"] = self._strategy_id
        info["strategy_id"] = self._strategy_id
        info["strategy_mode"] = "token"
        info["reward_raw"] = float(reward)
        info["reward_step"] = float(self.reward_per_step)
        info["reward_total_env"] = total_reward
        return self._obs(), total_reward, done, truncated, info

    def _apply_motif_edit(self, kind: str, value: str) -> None:
        assert self._motif_state is not None
        if kind in {"field", "field2", "price_field", "volume_field", "smooth_op"}:
            self._motif_state[kind] = value
            self._motif_edit_path.append(f"{kind}:{value}")
        elif kind == "window":
            self._motif_state[kind] = int(value)
            self._motif_edit_path.append(f"window:{value}")
        elif kind == "toggle":
            self._motif_state[value] = True
            self._motif_edit_path.append(f"add_{value}")
        self._motif_edit_count += 1

    def _finish_motif(self):
        assert self._motif_state is not None
        motif = self._current_motif()
        assert motif is not None
        try:
            expr = build_motif_expression(self._motif_state)
            natural = naturalness_score(expr)
            reward, info = self.pool.try_new_expr(
                expr,
                token_seq=None,
                source_head=self.current_strategy,
                motif_id=motif.motif_id,
                motif_family=motif.family,
                edit_path=list(self._motif_edit_path),
                naturalness_score=float(natural["score"]),
                naturalness_passed=bool(natural["passed"]),
                naturalness_reasons=list(natural["reasons"]),
                naturalness_stats=natural["stats"],
            )
        except OutOfDataRangeError:
            reward, info = 0.0, {"out_of_data": True}
        except Exception as exc:
            reward, info = -1.0, {"invalid": True, "error": str(exc)}

        info = dict(info)
        info.setdefault("source_head", self.current_strategy)
        info.setdefault("source_strategy", self.current_strategy)
        info.setdefault("motif_id", motif.motif_id)
        info.setdefault("motif_family", motif.family)
        info.setdefault("edit_path", list(self._motif_edit_path))
        info["head_id"] = self._strategy_id
        info["strategy_id"] = self._strategy_id
        info["strategy_mode"] = "motif"
        info["reward_raw"] = float(reward)
        info["reward_step"] = float(self.reward_per_step)
        info["reward_total_env"] = float(reward + self.reward_per_step)
        return self._obs(), float(reward + self.reward_per_step), True, False, info

    def _step_motif(self, action: int):
        if self._motif_state is None:
            self._motif_state = default_state(action)
            self._motif_edit_path = [f"motif:{MOTIF_BANK[action].motif_id}"]
            return self._obs(), float(self.reward_per_step), False, False, {
                "source_head": self.current_strategy,
                "source_strategy": self.current_strategy,
                "strategy_mode": "motif",
                "motif_id": MOTIF_BANK[action].motif_id,
                "motif_family": MOTIF_BANK[action].family,
            }

        if action < N_MOTIF_ACTIONS:
            old_motif = self._current_motif()
            self._motif_state = default_state(action)
            self._motif_edit_count += 1
            self._motif_edit_path = [
                f"motif:{MOTIF_BANK[action].motif_id}",
                f"switch_from:{old_motif.motif_id if old_motif is not None else 'none'}",
            ]
            return self._obs(), float(self.reward_per_step), False, False, {
                "source_head": self.current_strategy,
                "source_strategy": self.current_strategy,
                "strategy_mode": "motif",
                "motif_id": MOTIF_BANK[action].motif_id,
                "motif_family": MOTIF_BANK[action].family,
            }

        kind, value = EDIT_ACTIONS[action - N_MOTIF_ACTIONS]
        if kind == "stop":
            return self._finish_motif()
        self._apply_motif_edit(kind, value)
        return self._obs(), float(self.reward_per_step), False, False, {
            "source_head": self.current_strategy,
            "source_strategy": self.current_strategy,
            "strategy_mode": "motif",
            "edit_path": list(self._motif_edit_path),
        }

    def step(self, action: int):
        action = int(action)
        masks = self.action_masks()
        if action < 0 or action >= HYBRID_ACTIONS or not bool(masks[action]):
            return self._obs(), -1.0, True, False, {
                "invalid_action": action,
                "source_head": self.current_strategy,
                "source_strategy": self.current_strategy,
                "strategy_mode": self._mode,
            }
        if self._mode == "token":
            return self._step_token(action)
        return self._step_motif(action)
