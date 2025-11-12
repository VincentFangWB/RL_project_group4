from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.loader import download_ohlcv
from src.ta import build_features
from src.single_asset_env import SingleAssetTradingEnv, EnvConfig


def make_env(prices: pd.Series, features: pd.DataFrame, cfg: EnvConfig):
    def _init():
        return SingleAssetTradingEnv(prices=prices, features=features, config=cfg)
    return _init


def split_data(df: pd.DataFrame, train_end: str):
    train = df.loc[:train_end].copy()
    test = df.loc[train_end:].copy()
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--train-end", type=str, required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--save-dir", type=str, default="artifacts/ppo")
    args = parser.parse_args()

    raw = download_ohlcv(args.ticker, args.start, args.end)
    feats = build_features(raw)

    price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

    df_all = pd.concat({"price": price, "ret": feats["ret"]}, axis=1).dropna()
    feats = feats.loc[df_all.index]

    train_df, test_df = split_data(df_all, args.train_end)
    feats_train = feats.loc[train_df.index]

    cfg = EnvConfig(window=args.window, transaction_cost=args.transaction_cost)

    env = DummyVecEnv([make_env(prices=train_df["price"], features=feats_train, cfg=cfg)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "model.zip"))
    print(f"Model saved to {save_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
