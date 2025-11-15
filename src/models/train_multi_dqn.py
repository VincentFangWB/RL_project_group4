# src/models/train_multi_dqn.py

import argparse
from typing import List

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.data.multi_asset_loader import load_prices_matrix
from src.envs.multi_asset_portfolio_env import (
    MultiAssetConfig,
    MultiAssetPortfolioTemplateDQNEnv,
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
    Factory function that creates a single MultiAssetPortfolioTemplateDQNEnv instance.
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
        env = MultiAssetPortfolioTemplateDQNEnv(config=config, prices=prices)
        return env

    return _init


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN on multi-asset portfolio environment with template actions"
    )

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

    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=2000)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)

    parser.add_argument("--save-path", type=str, default="artifacts/dqn_multi_asset_template/model.zip")
    parser.add_argument("--tensorboard-log", type=str, default="tb_logs/dqn_multi_asset_template")

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
    ]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.save_path)

    vec_env.close()
    print(f"DQN (template actions) model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
