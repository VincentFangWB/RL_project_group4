# src/backtest/evaluate.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import mplfinance as mpf

from stable_baselines3 import PPO, A2C, SAC

from src.data.loader import download_ohlcv
from src.features.ta import build_features
from src.envs.single_asset_env import SingleAssetTradingEnv, EnvConfig
from src.utils import metrics as M

ALGOS = {"ppo": PPO, "a2c": A2C, "sac": SAC}

# ---------- column normalizers (same as before, compact) ----------
def _clean_key(s: str) -> str:
    return str(s).strip().lower().replace("_"," ").replace("-"," ")

def _flatten_columns_prioritize_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols = []
        for c in out.columns:
            parts = [str(x) for x in c]
            field = None
            for p in parts:
                k = _clean_key(p)
                if "adj" in k and "close" in k: field="Adj Close"; break
            if field is None:
                for p in parts:
                    k=_clean_key(p)
                    if k in ("open","high","low","close","volume"):
                        field={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}[k]
                        break
            cols.append(field if field else str(c[-1]))
        out.columns = cols
    else:
        cols=[]
        for c in out.columns:
            k=_clean_key(c)
            if "adj" in k and "close" in k: cols.append("Adj Close")
            elif "open" in k: cols.append("Open")
            elif "high" in k: cols.append("High")
            elif "low"  in k: cols.append("Low")
            elif "close" in k: cols.append("Close")
            elif "volume" in k: cols.append("Volume")
            else: cols.append(str(c))
        out.columns = cols
    return out

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        if "Adj Close" in df.columns: df["Close"]=df["Adj Close"]
        elif "Open" in df.columns: df["Close"]=df["Open"]
        else: raise ValueError("Cannot find Close/Adj Close/Open")
    for c in ("Open","High","Low"):
        if c not in df.columns: df[c]=df["Close"]
    if "Volume" not in df.columns: df["Volume"]=0
    return df[["Open","High","Low","Close","Volume"]].copy()

def _sanitize_ohlcv_for_mpf(df: pd.DataFrame) -> pd.DataFrame:
    data=_ensure_ohlcv(df)
    for c in data.columns: data[c]=pd.to_numeric(data[c], errors="coerce")
    data=data.dropna(subset=["Open","High","Low","Close"])
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index=pd.to_datetime(data.index, errors="coerce"); data=data[~data.index.isna()]
    return data

# ---------- plots ----------
def plot_equity_curves(out_dir: Path, rl_equity: pd.Series, bh_equity: pd.Series):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(rl_equity.index, rl_equity.values, label="RL NAV")
    ax.plot(bh_equity.index, bh_equity.values, label="Buy&Hold NAV")
    ax.set_title("Equity Curves (Test)"); ax.set_xlabel("Date"); ax.set_ylabel("NAV"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir/"equity_curve.png", dpi=150); plt.close(fig)

def plot_positions(out_dir: Path, positions: pd.Series):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(positions.index, positions.values, label="Target Position")
    ax.set_title("Position Over Time"); ax.set_xlabel("Date"); ax.set_ylabel("Position [-1,1]"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir/"positions.png", dpi=150); plt.close(fig)

def plot_kline_with_trades(out_dir: Path, ohlcv: pd.DataFrame, trades: pd.DataFrame,
                           lookback: int | None = 120, plot_start: str | None = None, plot_end: str | None = None):
    df = ohlcv.copy()
    if plot_start and plot_end: df=df.loc[plot_start:plot_end]
    elif lookback is not None and len(df)>lookback: df=df.iloc[-lookback:]
    tdf = trades.reindex(df.index)

    buy = pd.Series(np.nan, index=df.index); sell = pd.Series(np.nan, index=df.index)
    mb = (tdf["trade"]=="BUY") if "trade" in tdf.columns else pd.Series(False, index=df.index)
    ms = (tdf["trade"]=="SELL") if "trade" in tdf.columns else pd.Series(False, index=df.index)
    buy[mb.fillna(False)]  = tdf.loc[mb.fillna(False),"price"]
    sell[ms.fillna(False)] = tdf.loc[ms.fillna(False),"price"]

    if len(df)<2: return
    kwargs=dict(type="candle", volume=True, style="yahoo",
                savefig=dict(fname=str(out_dir/"kline_trades.png"), dpi=150, bbox_inches="tight"))
    ap=[]
    if buy.notna().any():  ap.append(mpf.make_addplot(buy,  type="scatter", markersize=60, marker="^"))
    if sell.notna().any(): ap.append(mpf.make_addplot(sell, type="scatter", markersize=60, marker="v"))
    if ap: kwargs["addplot"]=ap
    mpf.plot(df, **kwargs)

# ---------- rollout ----------
def rollout(model, env: SingleAssetTradingEnv, deterministic: bool=True):
    obs, info = env.reset()
    rows=[]; done=False; prev_pos=0.0
    while not done:
        action,_ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        t = env._t
        curr_date = env.features.index[t]
        pt = env.prices.iloc[t]
        price_t = float(pt.iloc[0] if isinstance(pt, pd.Series) else pt)
        rows.append({
            "date": curr_date, "price": price_t, "prev_pos": prev_pos,
            "action": float(np.asarray(action).reshape(-1)[0]),
            "new_pos": info["position"], "turnover": info.get("turnover", np.nan),
            "reward": float(reward), "portfolio_value": info.get("portfolio_value", np.nan),
            "trade": info.get("trade", "HOLD"),
        })
        prev_pos = info["position"]; done = terminated or truncated
    df = pd.DataFrame.from_records(rows).set_index("date")
    return df["reward"], df, df["new_pos"].rename("position")

def buy_and_hold_returns(prices: pd.Series) -> pd.Series:
    if isinstance(prices, pd.DataFrame): prices=prices.squeeze("columns")
    ret = prices.pct_change().shift(-1).fillna(0.0)
    return ret.iloc[1:].astype(float)

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--test-start", type=str, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--train-end", type=str, default=None)

    p.add_argument("--window", type=int, default=30)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--borrow-fee", type=float, default=0.0)

    p.add_argument("--action-mode", type=str, default="target", choices=["target","delta","discrete3"])
    p.add_argument("--min-pos", type=float, default=-1.0)
    p.add_argument("--max-pos", type=float, default= 1.0)

    # ===== NEW: reward vs benchmark =====
    p.add_argument("--reward-mode", type=str, default="active",
                   choices=["plain","active","active_norm"])
    p.add_argument("--benchmark", type=str, default=None,
                   help="Benchmark ticker; default=ticker for active/active_norm, None for plain")
    p.add_argument("--bm-weight", type=float, default=1.0,
                   help="w_bm in active reward (1.0 -> benchmark fully hedged)")
    p.add_argument("--ema-alpha", type=float, default=0.02, help="EMA alpha for active_norm")

    p.add_argument("--algo", type=str, default="ppo", choices=["ppo","a2c","sac"])
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="artifacts/eval")
    p.add_argument("--plot-lookback", type=int, default=120)
    p.add_argument("--plot-start", type=str, default=None)
    p.add_argument("--plot-end", type=str, default=None)
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # decide download start for warmup
    dl_start = args.start
    if args.test_start and args.start is None:
        ts = pd.Timestamp(args.test_start); dl_start = str((ts - BDay(args.window + 30)).date())

    # asset
    raw = download_ohlcv(args.ticker, dl_start or "1900-01-01", args.end)
    raw = _flatten_columns_prioritize_fields(raw)
    ohlcv = _ensure_ohlcv(raw.copy())
    if "Close" not in raw.columns: raw["Close"] = ohlcv["Close"]
    feats = build_features(raw)
    price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    if isinstance(price, pd.DataFrame): price = price.squeeze("columns")

    # benchmark
    bm_sym = args.benchmark
    if bm_sym is None and args.reward_mode != "plain":
        bm_sym = args.ticker  # beat the asset's own B&H by default
    bm_ret = None
    if bm_sym is not None:
        bm_raw = download_ohlcv(bm_sym, dl_start or "1900-01-01", args.end)
        bm_raw = _flatten_columns_prioritize_fields(bm_raw)
        bm_px = bm_raw["Adj Close"] if "Adj Close" in bm_raw.columns else bm_raw["Close"]
        if isinstance(bm_px, pd.DataFrame): bm_px = bm_px.squeeze("columns")
        bm_ret = bm_px.pct_change().shift(-1).fillna(0.0)

    # choose test slice
    if args.test_start:   test_idx = raw.loc[pd.Timestamp(args.test_start):].index
    elif args.train_end:  test_idx = raw.loc[args.train_end:].index
    else:                 test_idx = raw.index

    feats_test = feats.loc[test_idx]
    price_test = price.loc[test_idx]
    ohlcv_test = ohlcv.loc[test_idx]
    bm_ret_test = bm_ret.loc[test_idx] if bm_ret is not None else None

    cfg = EnvConfig(
        window=args.window,
        transaction_cost=args.transaction_cost,
        borrow_fee_annual=args.borrow_fee,
        action_mode=args.action_mode,
        min_pos=args.min_pos, max_pos=args.max_pos,
        include_position_in_obs=True,
        reward_mode=args.reward_mode,
        benchmark_weight=args.bm_weight,
        ema_alpha=args.ema_alpha,
    )
    env = SingleAssetTradingEnv(
        prices=price_test,
        features=feats_test.assign(ret=feats_test["ret"]),
        config=cfg,
        benchmark_returns=bm_ret_test,
    )

    Algo = ALGOS[args.algo]
    model = Algo.load(args.model_path)

    rewards, actions_df, positions = rollout(model, env, deterministic=not args.stochastic)
    bh_rets = buy_and_hold_returns(price_test).reindex(rewards.index).fillna(0.0)

    def summarize(name: str, r: pd.Series):
        arr = np.asarray(r).reshape(-1)
        return {
            "name": name,
            "annual_return": M.annualized_return(arr),
            "annual_vol": M.annualized_vol(arr),
            "sharpe": M.sharpe_ratio(arr),
            "max_drawdown": M.max_drawdown(arr),
            "cum_return": float(np.prod(1.0 + arr) - 1.0),
        }
    out_tbl = pd.DataFrame([summarize("RL", rewards), summarize("Buy&Hold", bh_rets)]).set_index("name")
    pd.set_option("display.float_format", lambda x: f"{x:,.2%}")
    print("\n=== Evaluation (Test) ==="); print(out_tbl)

    actions_df.to_csv(out_dir/"actions.csv"); actions_df[actions_df["trade"]!="HOLD"].to_csv(out_dir/"trades.csv")
    print(f"Saved operations to: {out_dir/'actions.csv'} and {out_dir/'trades.csv'}")

    rl_equity = (1.0 + rewards).cumprod(); bh_equity = (1.0 + bh_rets).cumprod()
    plot_equity_curves(out_dir, rl_equity, bh_equity)
    plot_positions(out_dir, positions)

    ohlcv_plot = _sanitize_ohlcv_for_mpf(ohlcv_test)
    if len(ohlcv_plot) >= 2:
        plot_kline_with_trades(out_dir, ohlcv_plot, actions_df,
                               lookback=args.plot_lookback, plot_start=args.plot_start, plot_end=args.plot_end)

if __name__ == "__main__":
    main()
