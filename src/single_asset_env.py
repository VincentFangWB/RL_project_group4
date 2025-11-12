from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    window: int = 30
    transaction_cost: float = 0.001  # per-unit turnover cost, e.g., 10 bps
    reward_scaling: float = 1.0
    seed: int | None = 42


class SingleAssetTradingEnv(gym.Env):
    """Single-asset continuous-position trading environment.

    Action: a_t ∈ [-1, 1] is *target* position (−1 fully short, +1 fully long).
    Observation: last `window` days of features flattened to 1D.
    Reward: r_t = position_{t-1} * return_t − cost * |position_t − position_{t-1}|.
    Portfolio value: V_t = V_{t-1} * (1 + r_t).
    """

    metadata = {"render_modes": []}

    def __init__(self, prices: pd.Series, features: pd.DataFrame, config: EnvConfig):
        assert prices.index.equals(features.index), "prices and features index must align"
        assert {"ret"}.issubset(features.columns), "features must include 'ret'"
        self.prices = prices.astype(np.float32)
        self.features = features.astype(np.float32)
        self.cfg = config

        self.window = config.window
        self.tc = float(config.transaction_cost)
        self.reward_scaling = float(config.reward_scaling)

        self._rng = np.random.default_rng(config.seed)

        feat_dim = self.features.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window * feat_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._start_idx = self.window
        self._end_idx = len(self.features) - 1

        self._t = None
        self._pos = None
        self._value = None
        self._ret_hist = None

    def _get_obs(self) -> np.ndarray:
        start = self._t - self.window
        window_feat = self.features.iloc[start:self._t].to_numpy(dtype=np.float32)
        return window_feat.reshape(-1)

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = self._start_idx
        self._pos = 0.0
        self._value = 1.0
        self._ret_hist = []  # store per-step portfolio returns
        obs = self._get_obs()
        info = {"position": self._pos, "portfolio_value": self._value, "t": self._t}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a = float(np.clip(action[0], -1.0, 1.0))
        day_ret = float(self.features.iloc[self._t]["ret"])  # simple return

        turnover = abs(a - self._pos)
        cost = self.tc * turnover

        reward = self._pos * day_ret - cost
        reward *= self.reward_scaling

        self._value *= (1.0 + reward)
        self._ret_hist.append(reward)

        # rebalancing happens end-of-day; hold `a` into next step
        self._pos = a
        self._t += 1

        terminated = self._t >= self._end_idx
        truncated = False

        obs = self._get_obs()
        info = {
            "position": self._pos,
            "portfolio_value": self._value,
            "t": self._t,
        }
        return obs, float(reward), terminated, truncated, info