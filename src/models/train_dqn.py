# src/models/train_dqn.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.data.loader import download_ohlcv
from src.features.ta import build_features
from src.envs.single_asset_env import SingleAssetTradingEnv, EnvConfig

# --- small helper: robust flatten if any library reintroduces MultiIndex
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(t[-1]).strip() for t in out.columns.to_flat_index()]
    return out

def make_env(prices: pd.Series, feats: pd.DataFrame, cfg: EnvConfig):
    # SB3's EvalCallback expects a Gym env; wrapping in Monitor gives episode stats
    return Monitor(SingleAssetTradingEnv(
        prices=prices,
        features=feats.assign(ret=feats["ret"]),
        config=cfg,
        benchmark_returns=None  # you can still pass bm returns if using reward_mode='active'
    ))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--train-end", type=str, required=True)
    p.add_argument("--window", type=int, default=30)

    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--borrow-fee", type=float, default=0.0)
    p.add_argument("--min-pos", type=float, default=-1.0)
    p.add_argument("--max-pos", type=float, default= 1.0)

    p.add_argument("--reward-mode", type=str, default="active", choices=["plain","active","active_norm"])
    p.add_argument("--benchmark", type=str, default=None)
    p.add_argument("--bm-weight", type=float, default=1.0)
    p.add_argument("--ema-alpha", type=float, default=0.02)

    # DQN hyperparams inspired by AWS preset: epsilon schedule, buffer, target sync, gamma=0.99
    # (see the blog’s preset & discussion of ε-greedy and target updates).
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--buffer-size", type=int, default=40_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--target-update", type=int, default=100)
    p.add_argument("--exploration-fraction", type=float, default=0.1)     # decay over first 10% of steps
    p.add_argument("--exploration-initial-eps", type=float, default=1.0)  # start fully exploring
    p.add_argument("--exploration-final-eps", type=float, default=0.01)   # end near-greedy

    p.add_argument("--advisor", type=str, default="none", choices=["none","gru"])
    p.add_argument("--advisor-window", type=int, default=10)
    p.add_argument("--advisor-epochs", type=int, default=5)

    p.add_argument("--save-dir", type=str, default="artifacts/dqn_model")
    p.add_argument("--eval-freq", type=int, default=10_000)  # periodic evaluation (like AWS preset)
    args = p.parse_args()

    # 1) data & features
    raw = _flatten_cols(download_ohlcv(args.ticker, args.start, args.end))
    feats = build_features(raw)
    price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    if isinstance(price, pd.DataFrame): price = price.squeeze("columns")

    # optional RNN advisor: append forecasted next-day return as a feature
    if args.advisor == "gru":
        from src.advisors.gru_advisor import fit_predict_gru
        adv = fit_predict_gru(price, window=args.advisor_window, epochs=args.advisor_epochs)
        feats = feats.join(adv.rename("adv_ret_pred")).fillna(0.0)

    # benchmark (only needed if reward_mode='active'/'active_norm')
    bm_sym = args.benchmark if args.benchmark else (args.ticker if args.reward_mode!="plain" else None)
    bm_ret=None
    if bm_sym is not None:
        bm_raw = _flatten_cols(download_ohlcv(bm_sym, args.start, args.end))
        bm_px = bm_raw["Adj Close"] if "Adj Close" in bm_raw.columns else bm_raw["Close"]
        if isinstance(bm_px, pd.DataFrame): bm_px = bm_px.squeeze("columns")
        bm_ret = bm_px.pct_change().shift(-1).fillna(0.0)

    # 2) splits
    train_idx = raw.loc[:args.train_end].index
    test_idx  = raw.loc[args.train_end:].index

    # 3) envs — DQN requires DISCRETE actions; we expose {-1,0,+1}
    cfg = EnvConfig(
        window=args.window,
        transaction_cost=args.transaction_cost,
        borrow_fee_annual=args.borrow_fee,
        action_mode="discrete3",
        min_pos=args.min_pos, max_pos=args.max_pos,
        include_position_in_obs=True,
        reward_mode=args.reward_mode,
        benchmark_weight=args.bm_weight,
        ema_alpha=args.ema_alpha,
    )

    train_env = make_env(price.loc[train_idx], feats.loc[train_idx], cfg)
    eval_env  = make_env(price.loc[test_idx],  feats.loc[test_idx],  cfg)

    # 4) model — ε-greedy replay DQN, with periodic evaluation (as in AWS post)
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
    )

    out = Path(args.save_dir); out.mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out),
        log_path=str(out),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)
    model.save(out / "model.zip")
    print(f"Saved: {out/'model.zip'}")

if __name__ == "__main__":
    main()
