# src/models/train_ppo.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from src.data.loader import download_ohlcv
from src.features.ta import build_features
from src.envs.single_asset_env import SingleAssetTradingEnv, EnvConfig

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols=[]
        for c in out.columns:
            k = str(c[-1]).lower()
            if "adj" in k and "close" in k: cols.append("Adj Close")
            elif "open" in k: cols.append("Open")
            elif "high" in k: cols.append("High")
            elif "low" in k: cols.append("Low")
            elif "close" in k: cols.append("Close")
            elif "volume" in k: cols.append("Volume")
            else: cols.append(str(c[-1]))
        out.columns=cols
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--train-end", type=str, required=True)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--borrow-fee", type=float, default=0.0)
    p.add_argument("--action-mode", type=str, default="target", choices=["target","delta","discrete3"])
    p.add_argument("--min-pos", type=float, default=-1.0)
    p.add_argument("--max-pos", type=float, default= 1.0)

    # NEW: reward vs benchmark
    p.add_argument("--reward-mode", type=str, default="active", choices=["plain","active","active_norm"])
    p.add_argument("--benchmark", type=str, default=None, help="default=ticker when reward-mode != plain")
    p.add_argument("--bm-weight", type=float, default=1.0)
    p.add_argument("--ema-alpha", type=float, default=0.02)

    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--save-dir", type=str, default="artifacts/ppo_model")
    args = p.parse_args()

    raw = _flatten(download_ohlcv(args.ticker, args.start, args.end))
    feats = build_features(raw)

    price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    if isinstance(price, pd.DataFrame): price = price.squeeze("columns")
    bm_sym = args.benchmark if args.benchmark else (args.ticker if args.reward_mode!="plain" else None)

    bm_ret = None
    if bm_sym is not None:
        bm_raw = _flatten(download_ohlcv(bm_sym, args.start, args.end))
        bm_px = bm_raw["Adj Close"] if "Adj Close" in bm_raw.columns else bm_raw["Close"]
        if isinstance(bm_px, pd.DataFrame): bm_px = bm_px.squeeze("columns")
        bm_ret = bm_px.pct_change().shift(-1).fillna(0.0)

    train_idx = raw.loc[:args.train_end].index
    env = SingleAssetTradingEnv(
        prices=price.loc[train_idx],
        features=feats.loc[train_idx].assign(ret=feats["ret"].loc[train_idx]),
        config=EnvConfig(
            window=args.window,
            transaction_cost=args.transaction_cost,
            borrow_fee_annual=args.borrow-fee if hasattr(args,'borrow-fee') else args.borrow_fee,
            action_mode=args.action_mode,
            min_pos=args.min_pos, max_pos=args.max_pos,
            include_position_in_obs=True,
            reward_mode=args.reward_mode,
            benchmark_weight=args.bm_weight,
            ema_alpha=args.ema_alpha,
        ),
        benchmark_returns=(bm_ret.loc[train_idx] if bm_ret is not None else None),
    )

    model = PPO("MlpPolicy", env, ent_coef=0.01, use_sde=(args.action_mode in ("target","delta")),
                sde_sample_freq=4, verbose=1).learn(total_timesteps=args.total_timesteps)

    out = Path(args.save_dir); out.mkdir(parents=True, exist_ok=True)
    model.save(out/"model.zip"); print(f"Saved: {out/'model.zip'}")

if __name__ == "__main__":
    main()
