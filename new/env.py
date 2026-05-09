from typing import Tuple

import gymnasium as gym
import numpy as np

from alphagen.config import *
from alphagen.data.expression import OutOfDataRangeError
from alphagen.data.tokens import *
from alphagen.rl.env.core import AlphaEnvCore
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


HEAD_NAMES = ("base", "trend", "volatility", "volume", "corr", "rank", "explore")
HEAD_TO_ID = {name: idx for idx, name in enumerate(HEAD_NAMES)}
STRATEGY_NAMES = HEAD_NAMES
STRATEGY_TO_ID = HEAD_TO_ID


class MultiHeadAlphaEnvCore(AlphaEnvCore):
    current_head: str = "base"

    def _evaluate(self):
        expr = self._builder.get_tree()
        if self._print_expr:
            print(expr)
        try:
            token_seq = [
                str(token)
                for token in self._tokens
                if not isinstance(token, SequenceIndicatorToken)
            ]
            try_new_expr = getattr(self.pool, "try_new_expr")
            ret, info = try_new_expr(expr, token_seq=token_seq, source_head=self.current_head)
            self.eval_cnt += 1
            info = dict(info)
            info.setdefault("source_strategy", self.current_head)
            return ret, info
        except OutOfDataRangeError:
            return 0.0, {
                "out_of_data": True,
                "source_head": self.current_head,
                "source_strategy": self.current_head,
            }


class MultiHeadAlphaEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env: MultiHeadAlphaEnvCore,
        method: str = "multihead_intrinsic",
        reward_per_step: float = REWARD_PER_STEP,
    ):
        super().__init__(env)
        self.env: MultiHeadAlphaEnvCore
        self.method = method
        self._reward_per_step = reward_per_step
        self._episode_idx = -1
        self._head_id = 0
        self.counter = 0
        high = np.full(MAX_EXPR_LENGTH + 1, SIZE_ALL - 1, dtype=np.uint8)
        high[-1] = len(HEAD_NAMES) - 1
        self.action_space = gym.spaces.Discrete(SIZE_ACTION)
        self.observation_space = gym.spaces.Box(low=0, high=high, shape=(MAX_EXPR_LENGTH + 1,), dtype=np.uint8)

    def _choose_head(self) -> int:
        if self.method == "single_transformer":
            return HEAD_TO_ID["base"]
        return (self._episode_idx % len(HEAD_NAMES))

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self._episode_idx += 1
        self._head_id = self._choose_head()
        self.env.current_head = HEAD_NAMES[self._head_id]
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH + 1, dtype=np.uint8)
        self.state[-1] = self._head_id
        self.env.reset()
        return self.state, {
            "source_head": self.env.current_head,
            "source_strategy": self.env.current_head,
            "head_id": self._head_id,
            "strategy_id": self._head_id,
        }

    def step(self, action: int):
        _, reward, done, truncated, info = self.env.step(action2token(action))
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        self.state[-1] = self._head_id
        total_reward = float(reward + self._reward_per_step)
        info = dict(info) if info is not None else {}
        info.setdefault("source_head", self.env.current_head)
        info.setdefault("source_strategy", self.env.current_head)
        info["head_id"] = self._head_id
        info["strategy_id"] = self._head_id
        info["reward_raw"] = float(reward)
        info["reward_step"] = float(self._reward_per_step)
        info["reward_total_env"] = total_reward
        return self.state, total_reward, done, truncated, info

    def action_masks(self) -> np.ndarray:
        res = np.zeros(SIZE_ACTION, dtype=bool)
        valid = self.env.valid_action_types()
        for i in range(OFFSET_OP, OFFSET_OP + SIZE_OP):
            if valid["op"][OPERATORS[i - OFFSET_OP].category_type()]:
                res[i - 1] = True
        if valid["select"][1]:
            for i in range(OFFSET_FEATURE, OFFSET_FEATURE + SIZE_FEATURE):
                res[i - 1] = True
        if valid["select"][2]:
            for i in range(OFFSET_CONSTANT, OFFSET_CONSTANT + SIZE_CONSTANT):
                res[i - 1] = True
        if valid["select"][3]:
            for i in range(OFFSET_DELTA_TIME, OFFSET_DELTA_TIME + SIZE_DELTA_TIME):
                res[i - 1] = True
        if valid["select"][4]:
            res[OFFSET_SEP - 1] = True
        return res


def MultiHeadAlphaEnv(pool, method: str, reward_per_step: float = REWARD_PER_STEP, **kwargs):
    return MultiHeadAlphaEnvWrapper(
        MultiHeadAlphaEnvCore(pool=pool, **kwargs),
        method=method,
        reward_per_step=reward_per_step,
    )
