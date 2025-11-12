from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.loader import download_ohlcv
from src.ta import build_features
from src.single_asset_env import SingleAssetTradingEnv, EnvConfig
from src import metrics as M


def rollout(model: PPO, env: SingleAssetTradingEnv) -> pd.Series:
    obs, info = env.reset()
    rets = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rets.append(reward)
        done = terminated or truncated
    return pd.Series(rets, dtype=float)


def buy_and_hold_returns(prices: pd.Series) -> pd.Series:
    # simple next-day return approximation
    ret = prices.pct_change().shift(-1).fillna(0.0)
    return ret.iloc[1:].astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--train-end", type=str, required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    raw = download_ohlcv(args.ticker, args.start, args.end)
    feats = build_features(raw)
    price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

    # test split
    test = raw.loc[args.train_end:]
    feats_test = feats.loc[test.index]
    price_test = price.loc[test.index]

    cfg = EnvConfig(window=args.window, transaction_cost=args.transaction_cost)
    env = SingleAssetTradingEnv(prices=price_test, features=feats_test.assign(ret=feats_test["ret"]), config=cfg)

    model = PPO.load(args.model_path)

    rl_rets = rollout(model, env)
    bh_rets = buy_and_hold_returns(price_test)

    n = min(len(rl_rets), len(bh_rets))
    rl_rets = rl_rets.iloc[:n]
    bh_rets = bh_rets.iloc[:n]

    def summarize(name: str, r: pd.Series):
        return {
            "name": name,
            "annual_return": M.annualized_return(r),
            "annual_vol": M.annualized_vol(r),
            "sharpe": M.sharpe_ratio(r),
            "max_drawdown": M.max_drawdown(r),
            "cum_return": float((1 + r).prod() - 1.0),
        }

    rl_summary = summarize("RL(PPO)", rl_rets)
    bh_summary = summarize("Buy&Hold", bh_rets)

    df = pd.DataFrame([rl_summary, bh_summary]).set_index("name")
    pd.set_option("display.float_format", lambda x: f"{x:,.2%}")
    print("\n=== Evaluation (Test) ===")
    print(df)


if __name__ == "__main__":
    main()