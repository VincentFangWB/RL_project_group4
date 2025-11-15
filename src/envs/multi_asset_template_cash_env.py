# src/envs/multi_asset_template_cash_env.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class MultiAssetTemplateCashConfig:
    """
    Config for template-action multi-asset env with explicit cash.

    We have N risky assets + 1 cash dimension in the weight vector, but only
    a small discrete set of "templates" that the agent can choose from.

    Typical templates:
        0: keep current weights
        1: equal-weight risky assets, 0% cash
        2: growth tilt (overweight some assets)
        3: defensive tilt
        4: high-cash (e.g. 70% cash, 30% equal-weight risky)
    """

    tickers: List[str]
    start: str
    end: str
    window: int
    transaction_cost: float
    risk_lambda: float
    rolling_window: int
    max_steps: int
    initial_nav: float = 1.0


class MultiAssetTemplateCashEnv(gym.Env):
    """
    Multi-asset portfolio env for Enhanced DQN.

    - Discrete action space (K templates).
    - Each action maps to a target weight vector over N risky assets + 1 cash.
    - Cash has zero return.
    - Reward is risk-aware with transaction costs.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: MultiAssetTemplateCashConfig, prices: np.ndarray, dates=None):
        super().__init__()

        self.config = config
        self.tickers = config.tickers
        self.prices = prices
        self.dates = dates
        self.num_assets = prices.shape[1]
        self.window = config.window

        # N assets + 1 cash
        self.num_weights = self.num_assets + 1

        # Define discrete action space: we will use 5 templates by default.
        # 0: keep current weights
        # 1: equal-weight risky assets (0% cash)
        # 2: growth tilt (overweight first half of assets)
        # 3: defensive tilt (overweight second half of assets)
        # 4: high cash (e.g. 70% cash, 30% equal-weight risky)
        self.num_actions = 5
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation: [window * num_assets returns] + [current weights] + [nav, step_fraction]
        obs_dim = self.window * self.num_assets + self.num_weights + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.initial_nav = config.initial_nav
        self.transaction_cost = config.transaction_cost

        self.current_idx = 0
        self.start_idx = self.window
        self.end_idx = prices.shape[0] - 1
        self.episode_length = min(config.max_steps, self.end_idx - self.start_idx)

        self.nav = self.initial_nav
        self.prev_nav = self.nav
        self.weights = np.zeros(self.num_weights, dtype=np.float32)
        self.current_step = 0
        self.returns_buffer: List[float] = []

        self._log_returns = self._compute_log_returns()

    # --------------------------------------------------------------
    # gym API
    # --------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.current_idx = self.start_idx
        self.current_step = 0
        self.nav = self.initial_nav
        self.prev_nav = self.nav
        self.returns_buffer = []

        # equal-weight risky, 0 cash
        self.weights = np.zeros(self.num_weights, dtype=np.float32)
        if self.num_assets > 0:
            self.weights[: self.num_assets] = 1.0 / float(self.num_assets)
        self.weights[-1] = 0.0

        obs = self._get_observation()
        info = {
            "nav": self.nav,
            "weights": self.weights.copy(),
        }
        return obs, info

    def step(self, action: int):
        target_weights = self._action_to_weights(action)

        if self.current_idx >= self.end_idx:
            obs = self._get_observation()
            info = {
                "nav": self.nav,
                "weights": self.weights.copy(),
                "truncated": True,
            }
            return obs, 0.0, True, False, info

        next_idx = self.current_idx + 1

        prices_t = self.prices[self.current_idx]
        prices_tp1 = self.prices[next_idx]
        asset_returns = prices_tp1 / (prices_t + 1e-8) - 1.0

        turnover = float(np.sum(np.abs(target_weights - self.weights)))
        cost = self.transaction_cost * turnover

        w_assets = target_weights[: self.num_assets]
        r_p = float(np.dot(w_assets, asset_returns))
        r_p_net = r_p - cost

        self.prev_nav = self.nav
        self.nav = self.nav * (1.0 + r_p_net)
        self.weights = target_weights
        self.current_idx = next_idx
        self.current_step += 1

        self._update_returns_buffer(r_p_net)
        reward = self._compute_reward(r_p_net)

        obs = self._get_observation()
        terminated = self.current_step >= self.episode_length
        truncated = False

        info = {
            "nav": self.nav,
            "prev_nav": self.prev_nav,
            "portfolio_return": r_p,
            "portfolio_return_net": r_p_net,
            "turnover": turnover,
            "cost": cost,
            "weights": self.weights.copy(),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.dates is not None and 0 <= self.current_idx < len(self.dates):
            date_str = str(self.dates[self.current_idx])
        else:
            date_str = f"t={self.current_idx}"

        print(f"[{date_str}] NAV={self.nav:.4f}, weights={self.weights}")

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _compute_log_returns(self) -> np.ndarray:
        eps = 1e-8
        return np.log(self.prices[1:] / (self.prices[:-1] + eps))

    def _get_observation(self) -> np.ndarray:
        start = self.current_idx - self.window
        end = self.current_idx

        if start < 0:
            start = 0
            end = start + self.window

        window_log_returns = self._log_returns[start:end]
        if window_log_returns.shape[0] < self.window:
            pad_len = self.window - window_log_returns.shape[0]
            pad = np.zeros((pad_len, self.num_assets), dtype=np.float32)
            window_log_returns = np.vstack([pad, window_log_returns])

        feature_returns = window_log_returns.flatten().astype(np.float32)

        step_fraction = 0.0
        if self.episode_length > 0:
            step_fraction = float(self.current_step) / float(self.episode_length)

        extra = np.array([self.nav, step_fraction], dtype=np.float32)

        obs = np.concatenate(
            [feature_returns, self.weights.astype(np.float32), extra],
            axis=0,
        )
        return obs

    def _action_to_weights(self, action: int) -> np.ndarray:
        """
        Map discrete action -> target weight vector over (assets + cash).

        Templates:
          0: keep current weights
          1: equal-weight risky, 0% cash
          2: growth tilt (overweight first half of assets)
          3: defensive tilt (overweight second half of assets)
          4: high-cash (e.g. 70% cash, 30% equal-weight risky)
        """
        if action == 0:
            return self.weights.copy().astype(np.float32)

        w = np.zeros(self.num_weights, dtype=np.float32)

        if self.num_assets == 0:
            w[-1] = 1.0  # all cash if no assets
            return w

        # equal-weight risky, 0% cash
        w[: self.num_assets] = 1.0 / float(self.num_assets)
        w[-1] = 0.0

        if action == 1:
            return w

        # growth = first half of assets
        half = max(1, self.num_assets // 2)
        growth_idx = list(range(0, half))
        defensive_idx = list(range(half, self.num_assets))

        if action == 2:
            # shift weight from defensive to growth
            alpha = 0.3
            w = self._tilt_bucket(w, bucket_idx=growth_idx, alpha=alpha)
            return w

        if action == 3:
            # shift weight from growth to defensive
            alpha = 0.3
            w = self._tilt_bucket(w, bucket_idx=defensive_idx, alpha=alpha)
            return w

        if action == 4:
            # high cash: e.g. 70% cash, 30% equal-weight risky
            risky_share = 0.3
            cash_share = 0.7
            w = np.zeros(self.num_weights, dtype=np.float32)
            w[: self.num_assets] = risky_share / float(self.num_assets)
            w[-1] = cash_share
            return w

        # fallback
        return w

    def _tilt_bucket(self, w: np.ndarray, bucket_idx: List[int], alpha: float) -> np.ndarray:
        """
        Shift alpha fraction of weight from non-bucket assets into the bucket.
        """
        w = w.copy()
        if len(bucket_idx) == 0:
            return w

        bucket_mask = np.zeros(self.num_assets, dtype=np.float32)
        bucket_mask[bucket_idx] = 1.0
        non_bucket_mask = 1.0 - bucket_mask

        asset_weights = w[: self.num_assets]
        bucket_weight = float(np.sum(asset_weights * bucket_mask))
        non_bucket_weight = float(np.sum(asset_weights * non_bucket_mask))

        if non_bucket_weight <= 0.0:
            return w

        shift = alpha * non_bucket_weight

        # reduce non-bucket weights proportionally
        non_bucket_weights = asset_weights * non_bucket_mask
        if np.sum(non_bucket_weights) > 0:
            non_bucket_weights *= (non_bucket_weight - shift) / (np.sum(non_bucket_weights) + 1e-8)

        # increase bucket weights proportionally
        bucket_weights = asset_weights * bucket_mask
        if np.sum(bucket_weights) > 0:
            bucket_weights *= (bucket_weight + shift) / (np.sum(bucket_weights) + 1e-8)

        new_asset_weights = non_bucket_weights + bucket_weights
        new_asset_weights = np.maximum(new_asset_weights, 0.0)
        new_asset_weights /= np.sum(new_asset_weights) + 1e-8

        w[: self.num_assets] = new_asset_weights
        # keep cash as-is (here 0). Caller can set cash separately if needed.
        return w

    def _update_returns_buffer(self, r_net: float):
        self.returns_buffer.append(float(r_net))
        if len(self.returns_buffer) > self.config.rolling_window:
            self.returns_buffer.pop(0)

    def _compute_reward(self, r_net: float) -> float:
        if self.config.risk_lambda <= 0.0:
            return float(r_net)
        if len(self.returns_buffer) < 2:
            return float(r_net)
        vol = float(np.std(self.returns_buffer))
        return float(r_net - self.config.risk_lambda * vol)
