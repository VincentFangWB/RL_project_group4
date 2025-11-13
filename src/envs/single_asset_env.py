# src/envs/single_asset_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    """
    Configuration for SingleAssetTradingEnv.
    """
    # Observation
    window: int = 30
    include_position_in_obs: bool = True

    # Actions
    # - "target": action in [-1,1] is the desired next target position
    # - "delta" : action in [-1,1] changes current position by action * step_size
    # - "discrete3": {-1, 0, +1} (short/flat/long)
    action_mode: str = "target"                   # "target" | "delta" | "discrete3"
    step_size: float = 0.25                        # used only for action_mode="delta"
    discrete_actions: Tuple[float, float, float] = (-1.0, 0.0, 1.0)

    # Position limits
    min_pos: float = -1.0
    max_pos: float = 1.0

    # Frictions & fees
    transaction_cost: float = 0.001                # linear cost per unit turnover (one-way)
    slippage_bps: float = 0.5                      # additional slippage cost per unit turnover (in bps)
    borrow_fee_annual: float = 0.0                 # annualized short borrow fee applied to |short exposure|

    # Reward shaping
    # - "plain"       : r_t = pos_{t-1} * ret_asset - costs
    # - "active"      : r_t = pos_{t-1} * ret_asset - w_bm * ret_bm - costs
    # - "active_norm" : same as active, divided by EMA std (risk-adjusted)
    reward_mode: str = "plain"                     # "plain" | "active" | "active_norm"
    benchmark_weight: float = 1.0                  # w_bm in active reward
    ema_alpha: float = 0.02                        # smoothing for active_norm
    eps: float = 1e-8                              # numerical epsilon
    clip_reward: Optional[float] = None            # if set, clip reward to [-clip, clip]

    # Episode control (optional)
    max_episode_steps: Optional[int] = None        # if set, triggers truncated=True when reached


class SingleAssetTradingEnv(gym.Env):
    """
    Long/short single-asset trading environment with realistic frictions and optional
    benchmark-aware reward.

    Key design choices:
    - **No look-ahead**: observation never contains `ret` (the next-period return used for reward).
    - **Execution convention**: action chosen at time t sets the *target position for t+1*.
      Reward at t is computed using the *previous* position (pos_{t-1}) and next-day asset return.
    - **Portfolio value**: always tracked using *portfolio* PnL (pos_{t-1} * asset_ret - costs),
      even when the learning reward is "active" (vs. a benchmark).

    Observation (flattened):
        concat of last `window` rows from feature matrix (excluding 'ret'), and
        optionally a constant column with current position.

    Action:
        - "target"    -> Box([-1,1]) target position
        - "delta"     -> Box([-1,1]) position += action * step_size
        - "discrete3" -> Discrete(3) mapped to {-1,0,+1}

    Reward:
        - "plain"      : pos_{t-1}*ret_asset - tc - borrow - slippage
        - "active"     : (pos_{t-1}*ret_asset - w_bm*ret_bm) - tc - borrow - slippage
        - "active_norm": active / EMA_std(active) - tc - borrow - slippage

    Info dict includes:
        position (float), portfolio_value (float), turnover (float), trade ("BUY"/"SELL"/"HOLD")
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        config: Optional[EnvConfig] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        super().__init__()
        self.cfg = config or EnvConfig()

        # --- sanitize inputs
        prices = prices.copy()
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze("columns")
        self.prices = pd.to_numeric(prices, errors="coerce").astype(float)
        self.prices.index = pd.to_datetime(self.prices.index, errors="coerce")
        self.prices = self.prices[~self.prices.index.isna()]

        feats = features.copy()
        if "ret" not in feats.columns:
            # next-day return label
            px = self.prices.reindex(feats.index).astype(float)
            feats["ret"] = px.pct_change().shift(-1).fillna(0.0)
        # keep only overlapping index, drop NaNs
        common_idx = self.prices.index.intersection(feats.index)
        feats = feats.loc[common_idx]
        self.prices = self.prices.loc[common_idx]
        feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # observation uses all columns EXCEPT 'ret'
        self._obs_cols = [c for c in feats.columns if c != "ret"]
        self.features = feats

        # benchmark returns (optional)
        self.bm_ret = None
        if benchmark_returns is not None:
            bm = benchmark_returns.copy()
            if isinstance(bm, pd.DataFrame):
                bm = bm.squeeze("columns")
            bm = pd.to_numeric(bm, errors="coerce").astype(float)
            bm.index = pd.to_datetime(bm.index, errors="coerce")
            self.bm_ret = bm.reindex(self.features.index).fillna(0.0)

        # --- action space
        if self.cfg.action_mode == "discrete3":
            self.action_space = spaces.Discrete(len(self.cfg.discrete_actions))
        else:
            # continuous [-1,1]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        # --- observation space
        n_base = len(self._obs_cols)
        n_feat = n_base + (1 if self.cfg.include_position_in_obs else 0)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.window * n_feat,),
            dtype=np.float32,
        )

        # runtime state
        self._t: int = 0
        self._steps: int = 0
        self.position: float = 0.0
        self.portfolio_value: float = 1.0
        self._obs: Optional[np.ndarray] = None

        # borrow fee per trading day
        self._borrow_daily = float(self.cfg.borrow_fee_annual) / 252.0

        # state for active_norm
        self._ema_mean = 0.0
        self._ema_m2 = 0.0
        self._ema_ready = False

        # precompute asset next-day returns for speed
        self._asset_ret = self.features["ret"].astype(float).values

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _clip_pos(self, x: float) -> float:
        return float(np.clip(x, self.cfg.min_pos, self.cfg.max_pos))

    def _window_obs(self, t: int) -> np.ndarray:
        """
        Build a flattened windowed observation ending at row t (inclusive),
        padding the beginning with the first row if needed. Excludes 'ret'.
        """
        start = t - self.cfg.window + 1
        mat = self.features[self._obs_cols].values
        if start < 0:
            pad = np.repeat(mat[[0], :], -start, axis=0)
            win = np.vstack([pad, mat[: t + 1, :]])
        else:
            win = mat[start : t + 1, :]

        if self.cfg.include_position_in_obs:
            pos_col = np.full((win.shape[0], 1), self.position, dtype=np.float32)
            win = np.hstack([win.astype(np.float32), pos_col])

        return win.astype(np.float32).reshape(-1)

    def _active_component(self, pos_prev: float, idx: int) -> float:
        """
        Compute the 'active' component (portfolio minus benchmark) *before* costs.
        """
        r_asset = float(self._asset_ret[idx])
        if self.cfg.reward_mode == "plain" or self.bm_ret is None:
            return pos_prev * r_asset
        r_bm = float(self.bm_ret.iloc[idx])
        return pos_prev * r_asset - self.cfg.benchmark_weight * r_bm

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        # start after we have 'window' rows
        self._t = self.cfg.window - 1
        self._steps = 0
        self.position = 0.0
        self.portfolio_value = 1.0

        # reset EMA stats for active_norm
        self._ema_mean = 0.0
        self._ema_m2 = 0.0
        self._ema_ready = False

        self._obs = self._window_obs(self._t)
        info = {"position": self.position, "portfolio_value": self.portfolio_value}
        return self._obs.copy(), info

    def step(self, action):
        # Map action -> target position for the NEXT step
        if self.cfg.action_mode == "discrete3":
            ai = int(action) if np.ndim(action) else int(action)
            ai = int(np.clip(ai, 0, len(self.cfg.discrete_actions) - 1))
            target_pos = float(self.cfg.discrete_actions[ai])
        elif self.cfg.action_mode == "delta":
            a = float(np.asarray(action).reshape(-1)[0])
            target_pos = self._clip_pos(self.position + a * self.cfg.step_size)
        else:  # "target"
            a = float(np.asarray(action).reshape(-1)[0])
            target_pos = self._clip_pos(a)

        # Costs based on turnover = |Δpos|
        dpos = target_pos - self.position
        turnover = abs(dpos)
        tc = float(self.cfg.transaction_cost) * turnover
        slippage = (self.cfg.slippage_bps / 10_000.0) * turnover

        # Borrow cost uses the *target* short exposure (approx)
        short_exposure = max(-target_pos, 0.0)
        borrow_cost = self._borrow_daily * short_exposure

        # Reward at t uses *previous* position
        idx = self._t
        pos_prev = self.position
        # asset next-day return
        r_asset = float(self._asset_ret[idx])

        # Active component
        active_before_costs = self._active_component(pos_prev, idx)

        # Risk-normalized active reward (optional)
        if self.cfg.reward_mode == "active_norm":
            x = active_before_costs
            if not self._ema_ready:
                self._ema_mean = x
                self._ema_m2 = x * x
                self._ema_ready = True
                core = 0.0  # first step has zero signal
            else:
                alpha = self.cfg.ema_alpha
                self._ema_mean = (1 - alpha) * self._ema_mean + alpha * x
                self._ema_m2 = (1 - alpha) * self._ema_m2 + alpha * (x * x)
                var = max(self._ema_m2 - self._ema_mean * self._ema_mean, 0.0)
                core = x / (np.sqrt(var) + self.cfg.eps)
            reward = core - tc - borrow_cost - slippage
        elif self.cfg.reward_mode == "active":
            reward = active_before_costs - tc - borrow_cost - slippage
        else:  # "plain"
            reward = (pos_prev * r_asset) - tc - borrow_cost - slippage

        # Optional reward clipping
        if self.cfg.clip_reward is not None:
            clip = float(self.cfg.clip_reward)
            reward = float(np.clip(reward, -clip, clip))

        # Portfolio value always tracks *portfolio* PnL (not the active reward)
        r_portfolio = (pos_prev * r_asset) - tc - borrow_cost - slippage
        self.portfolio_value *= (1.0 + r_portfolio)

        # Move position to target for next step
        self.position = float(target_pos)

        # Advance time
        self._t += 1
        self._steps += 1
        terminated = bool(self._t >= len(self.prices) - 1)
        truncated = bool(self.cfg.max_episode_steps is not None and self._steps >= self.cfg.max_episode_steps)

        # Next observation
        self._obs = self._window_obs(min(self._t, len(self.prices) - 1))

        trade = "HOLD"
        if dpos > 1e-9:
            trade = "BUY"
        elif dpos < -1e-9:
            trade = "SELL"

        info = {
            "position": float(self.position),
            "portfolio_value": float(self.portfolio_value),
            "turnover": float(turnover),
            "trade": trade,
        }
        return self._obs.copy(), float(reward), terminated, truncated, info

    # --------------------------------------------------------------------- #
    # Optional utilities
    # --------------------------------------------------------------------- #
    def render(self):
        return  # no-op; add custom plotting outside the env

    def close(self):
        return
