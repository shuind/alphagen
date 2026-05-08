from __future__ import annotations

from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from alphagen.config import REWARD_PER_STEP
from alphagen.data.expression import OutOfDataRangeError
from new.env import HEAD_NAMES, HEAD_TO_ID
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


EDIT_ACTIONS: Tuple[Tuple[str, str], ...] = (
    *((("field", name) for name in FIELD_NAMES)),
    *((("field2", name) for name in FIELD_NAMES)),
    *((("price_field", name) for name in PRICE_FIELDS)),
    *((("volume_field", name) for name in VOLUME_FIELDS)),
    *((("window", str(window)) for window in WINDOWS)),
    *((("smooth_op", name) for name in SMOOTH_OPS)),
    ("toggle", "rank_wrapper"),
    ("toggle", "ts_smooth"),
    ("toggle", "vol_adjust"),
    ("stop", "stop"),
)

N_MOTIF_ACTIONS = len(MOTIF_BANK)
N_ACTIONS = N_MOTIF_ACTIONS + len(EDIT_ACTIONS)
STOP_ACTION = N_ACTIONS - 1
OBS_DIM = 13


class MotifEditAlphaEnv(gym.Env):
    """Motif-first, edit-action AlphaGen environment.

    The action space is no longer the AlphaGen token vocabulary. Each episode selects
    one motif for the current round-robin head, applies a small number of semantic
    edits, then stops and evaluates the constructed expression.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pool,
        method: str = "motif_edit_intrinsic",
        max_edits: int = 4,
        reward_per_step: float = REWARD_PER_STEP,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.method = method
        self.max_edits = max(0, int(max_edits))
        self.reward_per_step = float(reward_per_step)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._episode_idx = -1
        self._head_id = 0
        self.current_head = "base"
        self._state: Dict[str, object] | None = None
        self._edit_count = 0
        self._edit_path: List[str] = []

    def _choose_head(self) -> int:
        return self._episode_idx % len(HEAD_NAMES)

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._episode_idx += 1
        self._head_id = self._choose_head()
        self.current_head = HEAD_NAMES[self._head_id]
        self._state = None
        self._edit_count = 0
        self._edit_path = []
        return self._obs(), {"source_head": self.current_head, "head_id": self._head_id}

    @property
    def unwrapped(self):  # type: ignore[override]
        return self

    def _obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = self._head_id / max(1, len(HEAD_NAMES) - 1)
        obs[1] = 1.0 if self._state is not None else 0.0
        if self._state is None:
            return obs

        motif_idx = int(self._state.get("motif_index", 0))
        obs[2] = motif_idx / max(1, len(MOTIF_BANK) - 1)
        obs[3] = self._edit_count / max(1, self.max_edits)
        obs[4] = FIELD_NAMES.index(str(self._state.get("field", "close"))) / max(1, len(FIELD_NAMES) - 1)
        obs[5] = FIELD_NAMES.index(str(self._state.get("field2", "open"))) / max(1, len(FIELD_NAMES) - 1)
        obs[6] = PRICE_FIELDS.index(str(self._state.get("price_field", "close"))) / max(1, len(PRICE_FIELDS) - 1)
        obs[7] = VOLUME_FIELDS.index(str(self._state.get("volume_field", "volume"))) / max(1, len(VOLUME_FIELDS) - 1)
        obs[8] = WINDOWS.index(int(self._state.get("window", 20))) / max(1, len(WINDOWS) - 1)
        obs[9] = SMOOTH_OPS.index(str(self._state.get("smooth_op", "mean"))) / max(1, len(SMOOTH_OPS) - 1)
        obs[10] = 1.0 if bool(self._state.get("rank_wrapper", False)) else 0.0
        obs[11] = 1.0 if bool(self._state.get("ts_smooth", False)) else 0.0
        obs[12] = 1.0 if bool(self._state.get("vol_adjust", False)) else 0.0
        return obs

    def _motif_is_valid_for_head(self, motif_index: int) -> bool:
        return motif_index in MOTIFS_BY_HEAD.get(self.current_head, [])

    def _current_motif(self):
        if self._state is None:
            return None
        return MOTIF_BANK[int(self._state["motif_index"])]

    def _edit_valid(self, kind: str, value: str) -> bool:
        if self._state is None or self._edit_count >= self.max_edits:
            return False
        motif = self._current_motif()
        if motif is None:
            return False
        if kind == "field":
            return motif.allow_field and value != self._state.get("field")
        if kind == "field2":
            return motif.allow_field2 and value != self._state.get("field2")
        if kind == "price_field":
            return motif.allow_price and value != self._state.get("price_field")
        if kind == "volume_field":
            return motif.allow_volume and value != self._state.get("volume_field")
        if kind == "window":
            return motif.allow_window and int(value) != int(self._state.get("window", 20))
        if kind == "smooth_op":
            return (motif.allow_smooth_op or bool(self._state.get("ts_smooth", False))) and value != self._state.get("smooth_op")
        if kind == "toggle" and value == "rank_wrapper":
            return motif.allow_rank_wrapper and not bool(self._state.get("rank_wrapper", False))
        if kind == "toggle" and value == "ts_smooth":
            return motif.allow_ts_smooth and not bool(self._state.get("ts_smooth", False))
        if kind == "toggle" and value == "vol_adjust":
            return motif.allow_vol_adjust and not bool(self._state.get("vol_adjust", False))
        return False

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if self._state is None:
            for idx in MOTIFS_BY_HEAD.get(self.current_head, []):
                mask[idx] = True
            return mask

        if self._edit_count >= self.max_edits:
            mask[STOP_ACTION] = True
            return mask

        # Same-head motif actions act as bounded "switch motif" edits.
        for idx in MOTIFS_BY_HEAD.get(self.current_head, []):
            if idx != int(self._state.get("motif_index", -1)):
                mask[idx] = True
        for offset, (kind, value) in enumerate(EDIT_ACTIONS):
            action = N_MOTIF_ACTIONS + offset
            if kind == "stop":
                mask[action] = True
            elif self._edit_valid(kind, value):
                mask[action] = True
        return mask

    def _apply_edit(self, kind: str, value: str) -> None:
        assert self._state is not None
        if kind in {"field", "field2", "price_field", "volume_field", "smooth_op"}:
            self._state[kind] = value
            self._edit_path.append(f"{kind}:{value}")
        elif kind == "window":
            self._state[kind] = int(value)
            self._edit_path.append(f"window:{value}")
        elif kind == "toggle":
            self._state[value] = True
            self._edit_path.append(f"add_{value}")
        self._edit_count += 1

    def _finish(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._state is not None
        motif = self._current_motif()
        assert motif is not None
        try:
            expr = build_motif_expression(self._state)
            natural = naturalness_score(expr)
            try_new_expr = getattr(self.pool, "try_new_expr")
            reward, info = try_new_expr(
                expr,
                token_seq=None,
                source_head=self.current_head,
                motif_id=motif.motif_id,
                motif_family=motif.family,
                edit_path=list(self._edit_path),
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
        info.setdefault("source_head", self.current_head)
        info.setdefault("motif_id", motif.motif_id)
        info.setdefault("motif_family", motif.family)
        info.setdefault("edit_path", list(self._edit_path))
        info["head_id"] = self._head_id
        info["reward_raw"] = float(reward)
        info["reward_step"] = float(self.reward_per_step)
        info["reward_total_env"] = float(reward + self.reward_per_step)
        return self._obs(), float(reward + self.reward_per_step), True, False, info

    def step(self, action: int):
        action = int(action)
        masks = self.action_masks()
        if action < 0 or action >= N_ACTIONS or not bool(masks[action]):
            return self._obs(), -1.0, True, False, {"invalid_action": int(action), "source_head": self.current_head}

        if self._state is None:
            self._state = default_state(action)
            self._edit_path = [f"motif:{MOTIF_BANK[action].motif_id}"]
            return self._obs(), float(self.reward_per_step), False, False, {
                "source_head": self.current_head,
                "motif_id": MOTIF_BANK[action].motif_id,
                "motif_family": MOTIF_BANK[action].family,
            }

        if action < N_MOTIF_ACTIONS:
            old_motif = self._current_motif()
            self._state = default_state(action)
            self._edit_count += 1
            self._edit_path = [
                f"motif:{MOTIF_BANK[action].motif_id}",
                f"switch_from:{old_motif.motif_id if old_motif is not None else 'none'}",
            ]
            return self._obs(), float(self.reward_per_step), False, False, {
                "source_head": self.current_head,
                "motif_id": MOTIF_BANK[action].motif_id,
                "motif_family": MOTIF_BANK[action].family,
            }

        kind, value = EDIT_ACTIONS[action - N_MOTIF_ACTIONS]
        if kind == "stop":
            return self._finish()
        self._apply_edit(kind, value)
        return self._obs(), float(self.reward_per_step), False, False, {
            "source_head": self.current_head,
            "edit_path": list(self._edit_path),
        }
