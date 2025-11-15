# src/models/train_multi_ppo.py

import argparse
from typing import List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from src.data.multi_asset_loader import load_prices_matrix
from src.envs.multi_asset_portfolio_env import (
    MultiAssetConfig,
    MultiAssetPortfolioEnv,
)


DEFAULT_TICKERS: List[str] = [
    "NVDA", "AAPL", "TSLA", "GS", "BAC",
    "XOM", "WMT", "KO", "UNH", "MCD",
]


def make_env(
    tickers: List[str],
    start: str,
    end: str,
    window: int,
    transaction_cost: float,
    risk_lambda: float,
    rolling_window: int,
    max_steps: int,
):
    """
    Factory function that creates a single MultiAssetPortfolioEnv instance.
    It is used for parallel vectorized environments.
    """

    def _init():
        prices, _ = load_prices_matrix(tickers=tickers, start=start, end=end)
        config = MultiAssetConfig(
            tickers=tickers,
            start=start,
            end=end,
            window=window,
            transaction_cost=transaction_cost,
            risk_lambda=risk_lambda,
            rolling_window=rolling_window,
            max_steps=max_steps,
        )
        env = MultiAssetPortfolioEnv(config=config, prices=prices)
        return env

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on multi-asset portfolio environment")

    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated list of tickers.",
    )
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--risk-lambda", type=float, default=0.1)
    parser.add_argument("--rolling-window", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=2000)

    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)

    parser.add_argument("--save-path", type=str, default="artifacts/ppo_multi_asset/model.zip")
    parser.add_argument("--tensorboard-log", type=str, default="tb_logs/ppo_multi_asset")

    return parser.parse_args()


def main():
    args = parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    env_fns = [
        make_env(
            tickers=tickers,
            start=args.start,
            end=args.end,
            window=args.window,
            transaction_cost=args.transaction_cost,
            risk_lambda=args.risk_lambda,
            rolling_window=args.rolling_window,
            max_steps=args.max_steps,
        )
        for _ in range(args.n_envs)
    ]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.batch_size // args.n_envs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.save_path)

    vec_env.close()
    print(f"PPO model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
