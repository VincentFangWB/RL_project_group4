from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class MultiAssetConfig:
    """
    Configuration for the multi-asset portfolio environments.
    """
    tickers: List[str]
    start: str
    end: str
    window: int = 30                 # Length of the historical window (in days).
    initial_cash: float = 1.0        # Initial portfolio net asset value (relative).
    transaction_cost: float = 0.001  # Transaction cost coefficient on turnover.
    risk_lambda: float = 0.1         # Risk penalty coefficient.
    rolling_window: int = 30         # Rolling window length for risk estimation.
    max_steps: Optional[int] = None  # Optional maximum number of steps per episode.


class MultiAssetPortfolioEnv(gym.Env):
    """
    Multi-asset portfolio management environment (continuous actions).

    Action space:
        R^{N+1} -> simplex via softmax. The first N components correspond to asset
        weights, and the last component is cash. The environment transforms raw
        actions into valid portfolio weights (non-negative and summing to one).

    Observation space:
        Concatenation of:
        - Recent window of log returns: shape [window * N]
        - Current weights: shape [N + 1]
        - Current NAV: shape [1]

    Reward:
        Risk-aware reward using a rolling estimate of volatility/std:
            reward_t = r_p_net_t - risk_lambda * rolling_std
        where r_p_net_t is the net portfolio return after transaction costs.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: MultiAssetConfig, prices: np.ndarray):
        """
        Parameters
        ----------
        config : MultiAssetConfig
            Environment configuration.
        prices : np.ndarray
            Price matrix with shape [T, N] (time, assets), aligned with config.tickers.
        """
        super().__init__()

        self.config = config
        self.tickers = config.tickers
        self.num_assets = len(self.tickers)

        # Price data: shape [T, N]
        self.prices = prices
        self.T = prices.shape[0]

        # Effective index range: start at `window` to ensure the initial window is available.
        self.start_idx = self.config.window
        self.end_idx = self.T - 1

        if self.config.max_steps is not None:
            self.episode_length = min(self.end_idx - self.start_idx, self.config.max_steps)
        else:
            self.episode_length = self.end_idx - self.start_idx

        # ====== Action space: continuous weights ======
        # Raw output in [-1, 1], will be transformed via softmax to weights on the simplex.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_assets + 1,),
            dtype=np.float32,
        )

        # ====== Observation space ======
        # - window * N log returns
        # - (N + 1) current weights
        # - 1 current NAV
        obs_dim = self.config.window * self.num_assets + (self.num_assets + 1) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ====== Internal state ======
        self.current_step: int = 0
        self.current_idx: int = 0  # Index in the underlying price array.
        self.nav: float = self.config.initial_cash
        self.prev_nav: float = self.nav

        # Weights: N assets + cash.
        self.weights: np.ndarray = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0  # Start fully in cash.

        # Rolling buffer of portfolio returns for risk estimation.
        self.returns_buffer: List[float] = []

    # ================== Gymnasium interface ==================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to the beginning of a new episode.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.current_idx = self.start_idx
        self.nav = self.config.initial_cash
        self.prev_nav = self.nav

        # Start with equal-weight allocation across all assets, no cash
        self.weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.weights[: self.num_assets] = 1.0 / float(self.num_assets)
        self.weights[-1] = 0.0

        self.returns_buffer = []

        obs = self._get_observation()
        info = {
            "nav": self.nav,
            "weights": self.weights.copy(),
        }
        return obs, info


    def step(self, action: np.ndarray):
        """
        Perform one environment step given a raw continuous action.

        Parameters
        ----------
        action : np.ndarray
            Raw action in R^{N+1} which will be mapped to valid weights.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # 1. Map raw action to valid weights on the simplex.
        target_weights = self._normalize_weights(action)

        # 2. Check if we have reached the end of the price series.
        if self.current_idx >= self.end_idx:
            obs = self._get_observation()
            info = {
                "nav": self.nav,
                "weights": self.weights.copy(),
                "truncated": True,
            }
            return obs, 0.0, True, False, info

        next_idx = self.current_idx + 1

        # 3. Compute asset returns: r_i = P_{t+1} / P_t - 1.
        prices_t = self.prices[self.current_idx]   # [N]
        prices_tp1 = self.prices[next_idx]         # [N]
        asset_returns = prices_tp1 / (prices_t + 1e-8) - 1.0  # [N]

        # 4. Turnover and transaction cost.
        turnover = float(np.sum(np.abs(target_weights - self.weights)))
        cost = self.config.transaction_cost * turnover

        # 5. Portfolio return (cash return is assumed to be zero).
        w_assets = target_weights[: self.num_assets]
        r_p = float(np.dot(w_assets, asset_returns))
        r_p_net = r_p - cost

        # Update NAV and state.
        self.prev_nav = self.nav
        self.nav = self.nav * (1.0 + r_p_net)
        self.weights = target_weights
        self.current_idx = next_idx
        self.current_step += 1

        # 6. Update rolling returns buffer and compute reward.
        self._update_returns_buffer(r_p_net)
        reward = self._compute_reward(r_p_net)

        # 7. Build next observation.
        obs = self._get_observation()

        # 8. Termination conditions.
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

    # ================== Helper methods ==================

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Map raw action vector to a valid portfolio weight vector on the simplex.

        The transformation uses a softmax so that all entries are non-negative and sum to one.
        """
        x = np.clip(action, -10.0, 10.0)
        e_x = np.exp(x - np.max(x))
        w = e_x / (np.sum(e_x) + 1e-8)
        return w.astype(np.float32)

    def _update_returns_buffer(self, r_p_net: float) -> None:
        """
        Update the rolling buffer of net portfolio returns.
        """
        self.returns_buffer.append(float(r_p_net))
        if len(self.returns_buffer) > self.config.rolling_window:
            self.returns_buffer.pop(0)

    def _compute_reward(self, r_p_net: float) -> float:
        """
        Compute the risk-aware reward.

        When the buffer is not long enough, return the net return.
        Once there is enough history, use:
            reward_t = r_p_net - risk_lambda * rolling_std
        where rolling_std is the standard deviation of returns in the buffer.
        """
        min_buffer = max(5, self.config.rolling_window // 3)
        if len(self.returns_buffer) < min_buffer:
            return float(r_p_net)

        mean_r = float(np.mean(self.returns_buffer))
        std_r = float(np.std(self.returns_buffer) + 1e-8)

        reward = r_p_net - self.config.risk_lambda * std_r

        # If you want Sharpe-like reward instead, you could use:
        # sharpe_t = mean_r / std_r
        # reward = sharpe_t

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector.

        Includes:
        - Log returns over the last `window` days: shape [window * N]
        - Current portfolio weights: shape [N + 1]
        - Current NAV: shape [1]
        """
        idx_end = self.current_idx
        idx_start = idx_end - self.config.window

        # Price window: shape [window + 1, N].
        prices_window = self.prices[idx_start: idx_end + 1]
        log_prices = np.log(prices_window + 1e-8)
        log_ret = log_prices[1:] - log_prices[:-1]  # [window, N]

        features = log_ret.reshape(-1)  # [window * N]

        obs = np.concatenate(
            [
                features,
                self.weights,          # [N + 1]
                np.array([self.nav]),  # [1]
            ],
            axis=0,
        ).astype(np.float32)

        return obs

    def render(self):
        """
        Print a simple summary of the current environment state.
        """
        print(
            f"[MultiAssetPortfolioEnv] step={self.current_step}, "
            f"idx={self.current_idx}, NAV={self.nav:.4f}"
        )

    def close(self):
        """
        Clean up internal resources (if needed).
        """
        pass


class MultiAssetPortfolioDiscreteEnv(MultiAssetPortfolioEnv):
    """
    Multi-asset portfolio management environment (discrete actions).

    This environment subclasses MultiAssetPortfolioEnv but overrides the action
    space and the step logic to interpret discrete actions.

    Action mapping:
        0: keep current weights (no rebalancing)
        1..N: fully long asset i (no cash)
        N+1: equal-weight all assets (no cash)

    Total number of actions = N + 2.
    """

    def __init__(self, config: MultiAssetConfig, prices: np.ndarray):
        super().__init__(config, prices)

        self.num_actions = self.num_assets + 2
        self.action_space = spaces.Discrete(self.num_actions)

    def step(self, action: int):
        """
        Step function for the discrete environment.

        Parameters
        ----------
        action : int
            Discrete action index.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # Map discrete action to target weights.
        if action == 0:
            # Keep current weights.
            target_weights = self.weights.copy()
        elif 1 <= action <= self.num_assets:
            # Fully allocate to a single asset.
            target_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
            target_weights[action - 1] = 1.0
        else:
            # Equal-weight all assets, no cash.
            target_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
            target_weights[: self.num_assets] = 1.0 / self.num_assets

        # Remaining logic is similar to the parent class, but without weight normalization.
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
        cost = self.config.transaction_cost * turnover

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

class MultiAssetPortfolioTemplateDQNEnv(MultiAssetPortfolioEnv):
    """
    Multi-asset portfolio management environment for DQN with template actions.

    Action mapping:
        0: keep current weights
        1: equal-weight all assets
        2: tilt towards growth bucket (e.g. NVDA, TSLA, AAPL)
        3: tilt towards defensive bucket (e.g. KO, WMT, XOM)
        4: tilt towards financial bucket (e.g. GS, BAC)

    Each action corresponds to a predefined target weight vector on the simplex.
    DQN only chooses among these few templates, which reduces extreme
    all-in/all-out moves and stabilizes learning.
    """

    def __init__(self, config: MultiAssetConfig, prices: np.ndarray):
        """
        Parameters
        ----------
        config : MultiAssetConfig
            Environment configuration.
        prices : np.ndarray
            Price matrix with shape [T, N].
        """
        super().__init__(config, prices)

        # Define discrete action space with 5 template actions.
        self.num_actions = 5
        self.action_space = spaces.Discrete(self.num_actions)

        # Pre-compute index sets for each bucket based on tickers.
        # We will build target weights using these buckets.
        self.growth_tickers = ["NVDA", "TSLA", "AAPL"]
        self.defensive_tickers = ["KO", "WMT", "XOM"]
        self.financial_tickers = ["GS", "BAC"]

        self.growth_idx = [i for i, t in enumerate(self.tickers) if t in self.growth_tickers]
        self.defensive_idx = [i for i, t in enumerate(self.tickers) if t in self.defensive_tickers]
        self.financial_idx = [i for i, t in enumerate(self.tickers) if t in self.financial_tickers]

        # Fallback in case some tickers are missing: use empty lists gracefully.
        # Actual weight computation will handle empty buckets.

    # --------------------------------------------------------------------- #
    # Overwrite step() to interpret discrete template actions as target
    # weight vectors, then reuse the parent class logic for P&L, costs,
    # risk-aware reward, etc.
    # --------------------------------------------------------------------- #
    def step(self, action: int):
        """
        Take one step in the environment given a discrete template action.

        Parameters
        ----------
        action : int
            Discrete action index in {0, 1, 2, 3, 4}.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # Map the discrete action to a target weight vector on the simplex.
        target_weights = self._action_to_weights(action)

        # The rest of the logic is almost identical to MultiAssetPortfolioEnv.step,
        # except that we do not call _normalize_weights() (weights already valid).
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
        cost = self.config.transaction_cost * turnover

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

    # --------------------------------------------------------------------- #
    # Helper: map action index -> template weights
    # --------------------------------------------------------------------- #
    def _action_to_weights(self, action: int) -> np.ndarray:
        """
        Map a discrete action to a target weight vector.

        The returned vector has shape [num_assets + 1] and lies on the simplex
        (non-negative entries summing to 1). The last component is cash.
        """
        # Start from current weights for action 0 (no rebalancing).
        if action == 0:
            return self.weights.copy().astype(np.float32)

        # Base: equal-weight all assets, no cash
        w = np.zeros(self.num_assets + 1, dtype=np.float32)
        w[: self.num_assets] = 1.0 / float(self.num_assets)
        w[-1] = 0.0

        # If there are no assets at all (should not happen), just keep current.
        if self.num_assets == 0:
            return self.weights.copy().astype(np.float32)

        # Tilt parameters: how much extra weight to allocate to the bucket.
        # We shift alpha percentage from "other assets" into the bucket.
        alpha = 0.3  # bucket tilt intensity in [0, 1)

        if action == 1:
            # Pure equal-weight across all assets (already set).
            return w

        # Choose which bucket to tilt towards.
        if action == 2:
            bucket_idx = self.growth_idx
        elif action == 3:
            bucket_idx = self.defensive_idx
        elif action == 4:
            bucket_idx = self.financial_idx
        else:
            # Unknown action, fall back to equal-weight.
            return w

        if len(bucket_idx) == 0:
            # If the bucket is empty (tickers not present), just return equal-weight.
            return w

        # Current equal-weight distribution for assets only.
        asset_weights = w[: self.num_assets]

        # Total weight of bucket and non-bucket assets.
        bucket_mask = np.zeros(self.num_assets, dtype=np.float32)
        bucket_mask[bucket_idx] = 1.0

        non_bucket_mask = 1.0 - bucket_mask

        bucket_weight = float(np.sum(asset_weights * bucket_mask))
        non_bucket_weight = float(np.sum(asset_weights * non_bucket_mask))

        if non_bucket_weight <= 0.0:
            # If by any chance there is no non-bucket weight, just return equal-weight.
            return w

        # We move alpha fraction of the non-bucket weight into the bucket.
        shift = alpha * non_bucket_weight

        # Reduce non-bucket weights proportionally.
        asset_weights = asset_weights * non_bucket_mask
        if np.sum(asset_weights) > 0:
            asset_weights *= (non_bucket_weight - shift) / (np.sum(asset_weights) + 1e-8)

        # Increase bucket weights proportionally to their current shares.
        bucket_weights = w[: self.num_assets] * bucket_mask
        if np.sum(bucket_weights) > 0:
            bucket_weights *= (bucket_weight + shift) / (np.sum(bucket_weights) + 1e-8)

        # Combine bucket and non-bucket components.
        new_asset_weights = asset_weights + bucket_weights

        # Safety normalization to ensure they sum to 1.
        new_asset_weights = np.maximum(new_asset_weights, 0.0)
        new_asset_weights /= np.sum(new_asset_weights) + 1e-8

        w[: self.num_assets] = new_asset_weights
        w[-1] = 0.0  # no cash in this template; you can add a "cash" template if desired

        return w.astype(np.float32)
