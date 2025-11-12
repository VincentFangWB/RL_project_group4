# src/envs/single_asset_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    # observation window
    window: int = 30
    # linear transaction cost per unit turnover (e.g., 0.001 = 10 bps one-way)
    transaction_cost: float = 0.001
    # action interpretation: "target" | "delta" | "discrete3"
    action_mode: str = "target"
    step_size: float = 0.1  # for "delta" mode
    discrete_actions: Tuple[float, float, float] = (-1.0, 0.0, 1.0)

    # long/short bounds
    min_pos: float = -1.0
    max_pos: float = 1.0

    # short borrow fee (annualized), charged daily on |short exposure|
    borrow_fee_annual: float = 0.0

    # include current position into observation
    include_position_in_obs: bool = True

    # ===== NEW: "beat-the-benchmark" reward =====
    # reward_mode: "plain" | "active" | "active_norm"
    reward_mode: str = "plain"
    benchmark_weight: float = 1.0  # w_bm in reward = pos*ret - w_bm*ret_bm - costs
    # EMA normalizer for active_norm
    ema_alpha: float = 0.02       # around 1/50 days
    eps: float = 1e-8


class SingleAssetTradingEnv(gym.Env):
    """
    Long/Short single-asset env with transaction costs, borrow fee, and optional
    benchmark-aware reward to directly optimize active return vs benchmark.

    Observation: last `window` rows of feature matrix (optionally + position), flattened.
    Action:
      - "target"  -> Box([-1,1]) gives target position directly
      - "delta"   -> Box([-1,1]) increments position by action * step_size
      - "discrete3" -> Discrete(3) => {-1, 0, +1}
    Reward:
      - plain : r_t = pos_{t-1}*ret_asset - tc - borrow
      - active: r_t = pos_{t-1}*ret_asset - w_bm*ret_bm - tc - borrow   (active return)
      - active_norm: r_t = (active return) / (EMA_std + eps)             (risk-adjusted)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        config: Optional[EnvConfig] = None,
        benchmark_returns: Optional[pd.Series] = None,  # NEW
    ):
        super().__init__()
        self.cfg = config or EnvConfig()

        # --- sanitize prices / features
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze("columns")
        self.prices = prices.astype(float).copy()
        self.features = features.copy()

        # ensure next-day return exists
        if "ret" not in self.features.columns:
            self.features["ret"] = self.prices.pct_change().shift(-1).fillna(0.0)

        # align, drop na
        self.features = self.features.dropna()
        self.prices = self.prices.loc[self.features.index]

        # --- benchmark returns (optional)
        self.bm_ret = None
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.DataFrame):
                benchmark_returns = benchmark_returns.squeeze("columns")
            benchmark_returns = benchmark_returns.astype(float)
            self.bm_ret = benchmark_returns.reindex(self.features.index).fillna(0.0)

        # --- spaces
        n_base = self.features.shape[1]
        self._use_pos = bool(self.cfg.include_position_in_obs)
        n_feat = n_base + (1 if self._use_pos else 0)

        if self.cfg.action_mode == "discrete3":
            self.action_space = spaces.Discrete(len(self.cfg.discrete_actions))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.window * n_feat,), dtype=np.float32
        )

        # --- state
        self._t = 0
        self.position = 0.0
        self.portfolio_value = 1.0

        # borrow/day approx (trading days)
        self._borrow_daily = float(self.cfg.borrow_fee_annual) / 252.0

        # for active_norm: EMA of mean/var of active returns
        self._ema_mean = 0.0
        self._ema_m2 = 0.0  # second moment
        self._ema_ready = False

        self._obs = None

    # ------------ helpers ------------
    def _clip_pos(self, x: float) -> float:
        return float(np.clip(x, self.cfg.min_pos, self.cfg.max_pos))

    def _get_window_obs(self, t: int) -> np.ndarray:
        start = t - self.cfg.window + 1
        if start < 0:
            pad = np.repeat(self.features.iloc[[0]].values, -start, axis=0)
            window = np.vstack([pad, self.features.iloc[: t + 1].values])
        else:
            window = self.features.iloc[start : t + 1].values

        if self._use_pos:
            pos_col = np.full((window.shape[0], 1), self.position, dtype=float)
            window = np.hstack([window, pos_col])

        return window.astype(np.float32).reshape(-1)

    def _active_return(self, pos_prev: float, ret_asset: float) -> float:
        # active return vs benchmark: pos_prev * ret_asset - w_bm * ret_bm
        if self.cfg.reward_mode == "plain" or self.bm_ret is None:
            return pos_prev * ret_asset
        ret_bm = float(self.bm_ret.iloc[self._t])
        return pos_prev * ret_asset - self.cfg.benchmark_weight * ret_bm

    # ------------ gymnasium API ------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._t = self.cfg.window - 1
        self.position = 0.0
        self.portfolio_value = 1.0
        self._ema_mean = 0.0
        self._ema_m2 = 0.0
        self._ema_ready = False
        self._obs = self._get_window_obs(self._t)
        return self._obs.copy(), {"position": self.position, "portfolio_value": self.portfolio_value}

    def step(self, action):
        # map action -> target position
        if self.cfg.action_mode == "discrete3":
            a = int(action) if np.ndim(action) else int(action)
            target_pos = float(self.cfg.discrete_actions[a])
        elif self.cfg.action_mode == "delta":
            a = float(np.asarray(action).reshape(-1)[0])
            target_pos = self._clip_pos(self.position + a * self.cfg.step_size)
        else:  # "target"
            a = float(np.asarray(action).reshape(-1)[0])
            target_pos = self._clip_pos(a)

        dpos = target_pos - self.position
        tc = self.cfg.transaction_cost * abs(dpos)

        ret_t = float(self.features.iloc[self._t]["ret"])
        short_exposure = max(-target_pos, 0.0)
        borrow_cost = self._borrow_daily * short_exposure

        # ===== reward building =====
        base_active = self._active_return(self.position, ret_t)  # uses pos_{t-1}
        if self.cfg.reward_mode == "active_norm":
            # EMA variance (Welford-style EMA)
            x = base_active
            if not self._ema_ready:
                self._ema_mean = x
                self._ema_m2 = x * x
                self._ema_ready = True
                norm = 0.0
            else:
                alpha = self.cfg.ema_alpha
                self._ema_mean = (1 - alpha) * self._ema_mean + alpha * x
                self._ema_m2 = (1 - alpha) * self._ema_m2 + alpha * (x * x)
                var = max(self._ema_m2 - self._ema_mean * self._ema_mean, 0.0)
                norm = x / (np.sqrt(var) + self.cfg.eps)
            reward = norm - tc - borrow_cost
        elif self.cfg.reward_mode == "active":
            reward = base_active - tc - borrow_cost
        else:  # "plain"
            reward = self.position * ret_t - tc - borrow_cost

        # update holdings to target for next day
        self.position = target_pos
        self.portfolio_value *= (1.0 + reward)

        # time moves
        self._t += 1
        terminated = self._t >= len(self.prices) - 1
        truncated = False

        self._obs = self._get_window_obs(self._t)
        info = {
            "position": self.position,
            "portfolio_value": self.portfolio_value,
            "turnover": abs(dpos),
            "trade": ("BUY" if dpos > 1e-8 else ("SELL" if dpos < -1e-8 else "HOLD")),
        }
        return self._obs.copy(), float(reward), bool(terminated), bool(truncated), info
