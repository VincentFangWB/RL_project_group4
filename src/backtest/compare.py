# src/backtest/compare.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from stable_baselines3 import PPO, A2C, SAC

from src.data.loader import download_ohlcv
from src.features.ta import build_features
from src.envs.single_asset_env import SingleAssetTradingEnv, EnvConfig
from src.utils import metrics as M

ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
}

# --------- helpers reused from evaluate.py ---------
def rollout(model, env: SingleAssetTradingEnv):
    """Deterministic rollout over test span.
    Returns: rewards(pd.Series), actions_df(pd.DataFrame), positions(pd.Series)
    """
    obs, info = env.reset()
    records = []
    done = False
    prev_pos = 0.0
    while not done:
        t = env._t
        curr_date = env.features.index[t]
        pt = env.prices.iloc[t]
        price_t = float(pt.iloc[0] if isinstance(pt, pd.Series) else pt)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        new_pos = float(info["position"])
        turnover = abs(new_pos - prev_pos)
        trade = "BUY" if new_pos > prev_pos + 1e-8 else ("SELL" if new_pos < prev_pos - 1e-8 else "HOLD")
        records.append({
            "date": curr_date,
            "price": price_t,
            "prev_pos": float(prev_pos),
            "action": float(np.asarray(action).reshape(-1)[0]),
            "new_pos": float(new_pos),
            "turnover": float(turnover),
            "reward": float(reward),
            "portfolio_value": float(info["portfolio_value"]),
            "trade": trade,
        })
        prev_pos = new_pos
        done = terminated or truncated

    actions_df = pd.DataFrame.from_records(records).set_index("date")
    rewards = actions_df["reward"]
    positions = actions_df["new_pos"].rename("position")
    return rewards, actions_df, positions


def buy_and_hold_returns(prices: pd.Series) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze("columns")
    ret = prices.pct_change().shift(-1).fillna(0.0)
    r = ret.iloc[1:].astype(float)
    r.index.name = prices.index.name
    return r


def plot_kline_with_trades(out_path: Path, ohlcv: pd.DataFrame, trades: pd.DataFrame,
                           lookback: int | None = 120,
                           plot_start: str | None = None, plot_end: str | None = None):
    df = ohlcv.copy()
    if plot_start and plot_end:
        df = df.loc[plot_start:plot_end]
    elif lookback is not None and len(df) > lookback:
        df = df.iloc[-lookback:]

    tdf = trades.reindex(df.index)

    buy_series = pd.Series(np.nan, index=df.index)
    sell_series = pd.Series(np.nan, index=df.index)
    mask_buy = (tdf["trade"] == "BUY") if "trade" in tdf.columns else pd.Series(False, index=df.index)
    mask_sell = (tdf["trade"] == "SELL") if "trade" in tdf.columns else pd.Series(False, index=df.index)
    buy_series[mask_buy.fillna(False)] = tdf.loc[mask_buy.fillna(False), "price"]
    sell_series[mask_sell.fillna(False)] = tdf.loc[mask_sell.fillna(False), "price"]

    kwargs = dict(
        type="candle",
        volume=True,
        style="yahoo",
        savefig=dict(fname=str(out_path), dpi=150, bbox_inches="tight"),
    )

    apds = []
    if buy_series.notna().any():
        apds.append(mpf.make_addplot(buy_series, type="scatter", markersize=60, marker="^"))
    if sell_series.notna().any():
        apds.append(mpf.make_addplot(sell_series, type="scatter", markersize=60, marker="v"))
    if apds:
        kwargs["addplot"] = apds  # mplfinance 不接受 addplot=None :contentReference[oaicite:2]{index=2}

    mpf.plot(df, **kwargs)

# --------- core evaluation for ONE algo ----------
def evaluate_one(algo: str, model_path: str,
                 price_test: pd.Series, feats_test: pd.DataFrame,
                 cfg: EnvConfig):
    AlgoClass = ALGOS[algo]
    model = AlgoClass.load(model_path)
    env = SingleAssetTradingEnv(prices=price_test, features=feats_test.assign(ret=feats_test["ret"]), config=cfg)

    rl_rets, actions_df, positions = rollout(model, env)
    bh_rets = buy_and_hold_returns(price_test).reindex(rl_rets.index).fillna(0.0)

    arr = np.asarray(rl_rets).reshape(-1)
    summary = {
        "algo": algo.upper(),
        "annual_return": M.annualized_return(arr),
        "annual_vol": M.annualized_vol(arr),
        "sharpe": M.sharpe_ratio(arr),
        "max_drawdown": M.max_drawdown(arr),
        "cum_return": float(np.prod(1.0 + arr) - 1.0),
    }
    rl_equity = (1.0 + rl_rets).cumprod()
    bh_equity = (1.0 + bh_rets).cumprod()
    return summary, actions_df, rl_equity, bh_equity, positions


def parse_model_map(items: list[str]) -> dict:
    """
    Parse --model like:  --model ppo=artifacts/ppo/model.zip --model a2c=... --model sac=...
    """
    out = {}
    for s in items:
        if "=" not in s:
            raise ValueError(f"--model expects algo=path, got: {s}")
        k, v = s.split("=", 1)
        k = k.strip().lower()
        out[k] = v.strip()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--train-end", type=str, required=True)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--out-dir", type=str, default="artifacts/compare")
    p.add_argument("--plot-lookback", type=int, default=120)
    p.add_argument("--plot-start", type=str, default=None)
    p.add_argument("--plot-end", type=str, default=None)
    p.add_argument("--algos", nargs="+", default=["ppo", "a2c", "sac"], choices=["ppo", "a2c", "sac"])
    p.add_argument("--model", action="append", default=[], help="Format: algo=path/to/model.zip (repeatable)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load data once
    raw = download_ohlcv(args.ticker, args.start, args.end)
    feats = build_features(raw)
    price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    if isinstance(price, pd.DataFrame):
        price = price.squeeze("columns")

    test = raw.loc[args.train_end:]
    feats_test = feats.loc[test.index]
    price_test = price.loc[test.index]

    cfg = EnvConfig(window=args.window, transaction_cost=args.transaction_cost)

    # 2) resolve models
    model_map = parse_model_map(args.model)  # dict: algo -> path
    for a in args.algos:
        if a not in model_map:
            raise SystemExit(f"Missing model path for '{a}'. Provide --model {a}=...</model.zip>")

    # 3) loop & collect
    metrics_rows = []
    equity_df = pd.DataFrame(index=price_test.index)  # we will join aligned equities
    combined_actions = []

    # Buy&Hold curve for overlay once
    bh_rets_all = buy_and_hold_returns(price_test)
    bh_equity_all = (1.0 + bh_rets_all).cumprod()
    equity_df["Buy&Hold"] = bh_equity_all

    for algo in args.algos:
        summary, actions_df, rl_equity, bh_equity, positions = evaluate_one(
            algo, model_map[algo], price_test, feats_test, cfg
        )

        # save per-algo artifacts
        (out_dir / algo).mkdir(parents=True, exist_ok=True)
        actions_df.to_csv(out_dir / algo / f"{algo}_actions.csv")
        actions_df[actions_df["trade"] != "HOLD"].to_csv(out_dir / algo / f"{algo}_trades.csv")

        # per-algo kline
        plot_kline_with_trades(
            out_path=out_dir / f"kline_{algo}.png",
            ohlcv=test[["Open", "High", "Low", "Close", "Volume"]],
            trades=actions_df,
            lookback=args.plot_lookback,
            plot_start=args.plot_start,
            plot_end=args.plot_end,
        )

        # collect for combined outputs
        metrics_rows.append(summary)
        equity_df[algo.upper()] = rl_equity.reindex(equity_df.index).ffill()
        tmp = actions_df.copy()
        tmp.insert(0, "algo", algo.upper())
        combined_actions.append(tmp)

    # 4) write combined tables
    metrics_df = pd.DataFrame(metrics_rows).set_index("algo")
    metrics_df.sort_index().to_csv(out_dir / "combined_metrics.csv")

    actions_all = pd.concat(combined_actions, axis=0, ignore_index=False)  # 合并动作表 :contentReference[oaicite:3]{index=3}
    actions_all.sort_index().to_csv(out_dir / "combined_actions.csv")
    actions_all[actions_all["trade"] != "HOLD"].to_csv(out_dir / "combined_trades.csv")

    # 5) plot combined equity
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in equity_df.columns:
        ax.plot(equity_df.index, equity_df[col].values, label=col)  # 同轴叠加多线条 :contentReference[oaicite:4]{index=4}
    ax.set_title(f"Equity Curves — {args.ticker} (Test)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Asset Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "combined_equity.png", dpi=150)
    plt.close(fig)

    # 6) console pretty print
    pd.set_option("display.float_format", lambda x: f"{x:,.2%}")
    print("\n=== Combined Metrics (Test) ===")
    print(metrics_df[["annual_return", "annual_vol", "sharpe", "max_drawdown", "cum_return"]])

if __name__ == "__main__":
    main()
