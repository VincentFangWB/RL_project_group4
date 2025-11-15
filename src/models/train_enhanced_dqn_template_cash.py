# src/models/train_enhanced_dqn_template_cash.py

import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.multi_asset_loader import load_prices_matrix
from src.envs.multi_asset_template_cash_env import (
    MultiAssetTemplateCashConfig,
    MultiAssetTemplateCashEnv,
)


# ---------- Dueling Q-Network ----------

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value_stream(f)          # [B, 1]
        a = self.adv_stream(f)            # [B, A]
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean
        return q


# ---------- Prioritized Replay ----------

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.eps = 1e-6

    def add(self, s, a, r, s_next, done):
        idx = self.pos
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s_next
        self.dones[idx] = float(done)

        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_probs(self):
        scaled = (self.priorities[: self.size] + self.eps) ** self.alpha
        total = scaled.sum()
        if total <= 0:
            return np.ones(self.size, dtype=np.float32) / float(self.size)
        return scaled / total

    def sample(self, batch_size: int, beta: float):
        assert self.size > 0
        probs = self._get_probs()
        idxs = np.random.choice(self.size, batch_size, p=probs)

        states = self.states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs]
        dones = self.dones[idxs]

        N = self.size
        weights = (N * probs[idxs]) ** (-beta)
        weights /= weights.max() + 1e-8
        weights = weights.astype(np.float32)

        return (states, actions, rewards, next_states, dones), weights, idxs

    def update_priorities(self, idxs, td_errors):
        td_errors = np.abs(td_errors) + self.eps
        for i, e in zip(idxs, td_errors):
            self.priorities[i] = float(e)


# ---------- Training Loop ----------

def train_enhanced_dqn_template_cash(
    env: MultiAssetTemplateCashEnv,
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
    obs_space = env.observation_space
    act_space = env.action_space
    assert len(obs_space.shape) == 1
    state_dim = obs_space.shape[0]
    action_dim = act_space.n

    device_t = torch.device(device)

    q_net = DuelingQNetwork(state_dim, action_dim).to(device_t)
    target_q = DuelingQNetwork(state_dim, action_dim).to(device_t)
    target_q.load_state_dict(q_net.state_dict())

    opt = optim.Adam(q_net.parameters(), lr=learning_rate)
    replay = PrioritizedReplayBuffer(buffer_size, state_dim, alpha=alpha)

    exploration_steps = int(exploration_fraction * total_timesteps)
    eps_start = 1.0
    eps_end = exploration_final_eps

    def get_eps(step: int) -> float:
        if step >= exploration_steps:
            return eps_end
        frac = step / float(exploration_steps)
        return eps_start + frac * (eps_end - eps_start)

    def get_beta(step: int) -> float:
        return beta_start + (beta_end - beta_start) * (step / float(total_timesteps))

    obs, info = env.reset()
    obs = np.asarray(obs, dtype=np.float32)

    global_step = 0

    while global_step < total_timesteps:
        eps = get_eps(global_step)

        if np.random.rand() < eps:
            action = act_space.sample()
        else:
            with torch.no_grad():
                s = torch.from_numpy(obs).unsqueeze(0).to(device_t)
                q = q_net(s)
                action = int(torch.argmax(q, dim=1).item())

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = terminated or truncated

        replay.add(obs, action, float(reward), next_obs, done)

        obs = next_obs
        global_step += 1

        if done:
            obs, info = env.reset()
            obs = np.asarray(obs, dtype=np.float32)

        if replay.size < start_learning_steps:
            continue

        if global_step % train_freq == 0:
            beta = get_beta(global_step)
            (s_b, a_b, r_b, s_next_b, d_b), w_b, idxs = replay.sample(batch_size, beta)

            s_t = torch.from_numpy(s_b).to(device_t)
            a_t = torch.from_numpy(a_b).to(device_t)
            r_t = torch.from_numpy(r_b).to(device_t)
            s_next_t = torch.from_numpy(s_next_b).to(device_t)
            d_t = torch.from_numpy(d_b).to(device_t)
            w_t = torch.from_numpy(w_b).to(device_t)

            q_vals = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next_online = q_net(s_next_t)
                next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)
                q_next_target = target_q(s_next_t).gather(1, next_actions).squeeze(1)
                target = r_t + gamma * (1.0 - d_t) * q_next_target

            td_errors = target - q_vals
            loss = (w_t * td_errors.pow(2)).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
            opt.step()

            replay.update_priorities(idxs, td_errors.detach().cpu().numpy())

        if global_step % target_update_interval == 0:
            target_q.load_state_dict(q_net.state_dict())
            print(f"[step {global_step}] target network updated.")

    return q_net, target_q


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Enhanced DQN (Double-Dueling + PER) on template-action multi-asset env with cash."
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

    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-learning-steps", type=int, default=10_000)
    parser.add_argument("--target-update-interval", type=int, default=2_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)

    parser.add_argument("--save-path", type=str, default="artifacts/enhanced_dqn_template_cash/model.pt")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    prices, dates = load_prices_matrix(tickers=tickers, start=args.start, end=args.end)

    cfg = MultiAssetTemplateCashConfig(
        tickers=tickers,
        start=args.start,
        end=args.end,
        window=args.window,
        transaction_cost=args.transaction_cost,
        risk_lambda=args.risk_lambda,
        rolling_window=args.rolling_window,
        max_steps=args.max_steps,
    )
    env = MultiAssetTemplateCashEnv(config=cfg, prices=prices, dates=dates)

    q_net, target_q = train_enhanced_dqn_template_cash(
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

    torch.save(q_net.state_dict(), args.save_path)
    print(f"Enhanced DQN (template+cash) model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
