# src/models/train_multi_enhanced_dqn.py

import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.multi_asset_loader import load_prices_matrix
from src.envs.multi_asset_portfolio_env import (
    MultiAssetConfig,
    MultiAssetPortfolioTemplateDQNEnv,
)


# ============================================================
# 1. Dueling Q-Network (for Double DQN)
# ============================================================

class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network with separate value and advantage streams.

    Architecture:
        feature -> (value_head, advantage_head)
        Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: outputs scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream: outputs A(s,a) for each action
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape [batch_size, state_dim].

        Returns
        -------
        q_values : torch.Tensor
            Q-values for each action, shape [batch_size, action_dim].
        """
        features = self.feature(x)
        value = self.value_stream(features)                 # [B, 1]
        adv = self.adv_stream(features)                     # [B, A]

        # Subtract mean advantage to make Q identifiable.
        adv_mean = adv.mean(dim=1, keepdim=True)
        q_values = value + adv - adv_mean
        return q_values


# ============================================================
# 2. Prioritized Experience Replay (proportional variant)
#    Based on Schaul et al., 2015 "Prioritized Experience Replay"
# ============================================================

class PrioritizedReplayBuffer:
    """
    Proportional prioritized replay buffer.

    Each transition has a priority p_i; sampling probability is:
        P(i) = p_i^alpha / sum_j p_j^alpha

    Importance sampling weight:
        w_i = (N * P(i))^-beta  /  max_j w_j

    This implementation uses a simple array-based segment tree.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
    ):
        self.capacity = capacity
        self.alpha = alpha

        self.pos = 0
        self.size = 0

        # Experience storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        # Priorities (we store raw priorities, not probabilities)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.epsilon = 1e-6  # small constant to avoid zero priority

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a new transition to the buffer.
        Priority is initialized with max priority so that new samples
        are likely to be sampled at least once.
        """
        idx = self.pos

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_probabilities(self) -> np.ndarray:
        """
        Compute sampling probabilities for all stored transitions.
        """
        scaled_prios = (self.priorities[: self.size] + self.epsilon) ** self.alpha
        total = scaled_prios.sum()
        if total <= 0.0:
            return np.ones(self.size, dtype=np.float32) / float(self.size)
        return scaled_prios / total

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions with importance sampling weights.

        Returns
        -------
        batch : tuple
            (states, actions, rewards, next_states, dones)
        weights : np.ndarray
            Importance sampling weights, shape [batch_size].
        indices : np.ndarray
            Indices of sampled transitions, used to update priorities.
        """
        assert self.size > 0, "Cannot sample from an empty buffer."

        probs = self._get_probabilities()
        indices = np.random.choice(self.size, batch_size, p=probs)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        # Importance sampling weights
        N = self.size
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-8
        weights = weights.astype(np.float32)

        batch = (states, actions, rewards, next_states, dones)
        return batch, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled transitions based on absolute TD-errors.

        Parameters
        ----------
        indices : np.ndarray
            Indices of transitions to update.
        td_errors : np.ndarray
            TD-errors of shape [batch_size].
        """
        td_errors = np.abs(td_errors) + self.epsilon
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(err)


# ============================================================
# 3. Training loop: Double-Dueling DQN + PER
# ============================================================

def train_enhanced_dqn(
    env: MultiAssetPortfolioTemplateDQNEnv,
    total_timesteps: int = 1_000_000,
    learning_rate: float = 5e-5,
    gamma: float = 0.98,
    buffer_size: int = 500_000,
    batch_size: int = 128,
    start_learning_steps: int = 10_000,
    target_update_interval: int = 2_000,
    train_freq: int = 1,
    alpha: float = 0.6,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    exploration_fraction: float = 0.2,
    exploration_final_eps: float = 0.05,
    device: str = "cpu",
):
    """
    Main training loop for Double-Dueling DQN with prioritized replay.

    Parameters
    ----------
    env : MultiAssetPortfolioTemplateDQNEnv
        The trading environment with discrete template actions.
    total_timesteps : int
        Total number of environment steps to run.
    learning_rate : float
        Optimizer learning rate.
    gamma : float
        Discount factor.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Batch size for gradient updates.
    start_learning_steps : int
        Number of steps to collect before starting gradient updates.
    target_update_interval : int
        How often (in steps) to copy online network weights to target network.
    train_freq : int
        Perform a gradient update every `train_freq` environment steps.
    alpha : float
        PER exponent for priorities.
    beta_start : float
        Initial importance-sampling exponent.
    beta_end : float
        Final importance-sampling exponent (linearly annealed).
    exploration_fraction : float
        Fraction of total steps over which epsilon is annealed from 1.0 to
        exploration_final_eps.
    exploration_final_eps : float
        Final epsilon for epsilon-greedy exploration.
    device : str
        "cpu" or "cuda".
    """
    obs_space = env.observation_space
    act_space = env.action_space

    assert len(obs_space.shape) == 1, "This implementation expects a flat observation vector."
    state_dim = obs_space.shape[0]
    action_dim = act_space.n

    device = torch.device(device)

    # Online and target networks
    q_net = DuelingQNetwork(state_dim, action_dim).to(device)
    target_q_net = DuelingQNetwork(state_dim, action_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    # Prioritized replay buffer
    replay_buffer = PrioritizedReplayBuffer(
        capacity=buffer_size,
        state_dim=state_dim,
        alpha=alpha,
    )

    # Epsilon-greedy schedule
    exploration_steps = int(exploration_fraction * total_timesteps)
    eps_start = 1.0
    eps_final = exploration_final_eps

    def get_epsilon(step: int) -> float:
        if step >= exploration_steps:
            return eps_final
        frac = step / float(exploration_steps)
        return eps_start + frac * (eps_final - eps_start)

    # Beta schedule for PER
    def get_beta(step: int) -> float:
        return beta_start + (beta_end - beta_start) * (step / float(total_timesteps))

    # Training loop
    state, info = env.reset()
    state = np.asarray(state, dtype=np.float32)

    global_step = 0

    while global_step < total_timesteps:
        epsilon = get_epsilon(global_step)

        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = act_space.sample()
        else:
            with torch.no_grad():
                s_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                q_values = q_net(s_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

        next_state, reward, terminated, truncated, step_info = env.step(action)
        next_state = np.asarray(next_state, dtype=np.float32)
        done = terminated or truncated

        replay_buffer.add(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=done,
        )

        state = next_state
        global_step += 1

        # Reset environment when episode ends
        if done:
            state, info = env.reset()
            state = np.asarray(state, dtype=np.float32)

        # Start learning after enough data has been collected
        if replay_buffer.size < start_learning_steps:
            continue

        # Perform gradient updates
        if global_step % train_freq == 0:
            beta = get_beta(global_step)
            (states_b, actions_b, rewards_b, next_states_b, dones_b), weights_b, indices_b = \
                replay_buffer.sample(batch_size=batch_size, beta=beta)

            states_t = torch.from_numpy(states_b).to(device)
            actions_t = torch.from_numpy(actions_b).to(device)
            rewards_t = torch.from_numpy(rewards_b).to(device)
            next_states_t = torch.from_numpy(next_states_b).to(device)
            dones_t = torch.from_numpy(dones_b).to(device)
            weights_t = torch.from_numpy(weights_b).to(device)

            # Current Q estimates
            q_values = q_net(states_t)                             # [B, A]
            q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Double DQN target:
            # a* = argmax_a Q_online(s', a)
            with torch.no_grad():
                next_q_online = q_net(next_states_t)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

                next_q_target = target_q_net(next_states_t)
                next_q_values = next_q_target.gather(1, next_actions).squeeze(1)

                target_q = rewards_t + gamma * (1.0 - dones_t) * next_q_values

            # TD error and PER weights
            td_errors = target_q - q_values
            loss = (weights_t * td_errors.pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
            optimizer.step()

            # Update priorities in replay buffer
            td_errors_np = td_errors.detach().cpu().numpy()
            replay_buffer.update_priorities(indices_b, td_errors_np)

        # Periodically update target network
        if global_step % target_update_interval == 0:
            target_q_net.load_state_dict(q_net.state_dict())
            print(f"[step {global_step}] Target network updated.")

    return q_net, target_q_net


# ============================================================
# 4. CLI wrapper for your project
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Double-Dueling DQN with PER on multi-asset portfolio env."
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default="NVDA,AAPL,TSLA,GS,BAC,XOM,WMT,KO,UNH,MCD",
        help="Comma-separated list of tickers.",
    )
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--risk-lambda", type=float, default=0.1)
    parser.add_argument("--rolling-window", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=2000)

    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-learning-steps", type=int, default=10_000)
    parser.add_argument("--target-update-interval", type=int, default=2000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)

    parser.add_argument("--save-path", type=str, default="artifacts/enhanced_dqn/model.pt")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    # Build prices & env
    prices, _ = load_prices_matrix(tickers=tickers, start=args.start, end=args.end)
    config = MultiAssetConfig(
        tickers=tickers,
        start=args.start,
        end=args.end,
        window=args.window,
        transaction_cost=args.transaction_cost,
        risk_lambda=args.risk_lambda,
        rolling_window=args.rolling_window,
        max_steps=args.max_steps,
    )
    env = MultiAssetPortfolioTemplateDQNEnv(config=config, prices=prices)

    q_net, target_q_net = train_enhanced_dqn(
        env=env,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        start_learning_steps=args.start_learning_steps,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        alpha=args.alpha,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        device=args.device,
    )

    # Save only the online network; evaluation can rebuild policy from it.
    torch.save(q_net.state_dict(), args.save_path)
    print(f"Enhanced DQN model (Double-Dueling + PER) saved to: {args.save_path}")


if __name__ == "__main__":
    main()
