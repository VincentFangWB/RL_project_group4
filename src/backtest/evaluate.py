# src/backtest/evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import mplfinance as mpf

from stable_baselines3 import PPO, A2C, SAC, DQN

from src.data.loader import download_ohlcv
from src.features.ta import build_features
from src.envs.single_asset_env import SingleAssetTradingEnv, EnvConfig
from src.utils import metrics as M

from src.advisors.gru_advisor import fit_predict_gru



ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "dqn": DQN,
}


# ----------------------------------------------------------------------
# helpers for price/ohlcv handling
# ----------------------------------------------------------------------
def _clean_key(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def _flatten_columns_prioritize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any MultiIndex or weird column names into something like:
    Open, High, Low, Close, Adj Close, Volume.
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        new_cols = []
        for col in out.columns:
            parts = [str(x) for x in col]
            field = None
            # prefer Adj Close if any part looks like it
            for p in parts:
                k = _clean_key(p)
                if "adj" in k and "close" in k:
                    field = "Adj Close"
                    break
            if field is None:
                for p in parts:
                    k = _clean_key(p)
                    if k in ("open", "high", "low", "close", "volume"):
                        field = {
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        }[k]
                        break
            new_cols.append(field if field else str(col[-1]))
        out.columns = new_cols
    else:
        new_cols = []
        for c in out.columns:
            k = _clean_key(c)
            if "adj" in k and "close" in k:
                new_cols.append("Adj Close")
            elif "open" in k:
                new_cols.append("Open")
            elif "high" in k:
                new_cols.append("High")
            elif "low" in k:
                new_cols.append("Low")
            elif "close" in k:
                new_cols.append("Close")
            elif "volume" in k:
                new_cols.append("Volume")
            else:
                new_cols.append(str(c))
        out.columns = new_cols

    return out


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have numeric OHLCV columns for plotting/backtesting.
    If Open/High/Low are missing, we copy Close.
    """
    out = df.copy()
    if "Close" not in out.columns:
        if "Adj Close" in out.columns:
            out["Close"] = out["Adj Close"]
        elif "Open" in out.columns:
            out["Close"] = out["Open"]
        else:
            raise ValueError("Cannot find Close/Adj Close/Open in columns")

    for c in ("Open", "High", "Low"):
        if c not in out.columns:
            out[c] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    cols = ["Open", "High", "Low", "Close", "Volume"]
    return out[cols].copy()


def _sanitize_ohlcv_for_mpf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare OHLCV data for mplfinance: numeric columns + DatetimeIndex.
    """
    data = _ensure_ohlcv(df)
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="coerce")
        data = data[~data.index.isna()]
    return data


# ----------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------
def plot_equity_curves(out_dir: Path, rl_equity: pd.Series, bh_equity: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rl_equity.index, rl_equity.values, label="Portfolio NAV")
    ax.plot(bh_equity.index, bh_equity.values, label="Buy&Hold NAV")
    ax.set_title("Equity Curves (Portfolio vs Buy&Hold)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve_portfolio.png", dpi=150)
    plt.close(fig)


def plot_active_equity(out_dir: Path, active_equity: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(active_equity.index, active_equity.values, label="Active NAV")
    ax.axhline(1.0, ls="--", lw=1, label="Benchmark")
    ax.set_title("Active NAV (Excess Return vs Benchmark)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Excess NAV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve_active.png", dpi=150)
    plt.close(fig)


def plot_accumulated_rewards(out_dir: Path, rewards: pd.Series, reward_mode: str = "plain"):
    fig, ax = plt.subplots(figsize=(10, 4))
    if reward_mode in ("plain", "active"):
        y = (1.0 + rewards).cumprod() - 1.0
        ax.set_ylabel("Cumulative Return from Reward")
        label = "Cumulative Return (from reward)"
    else:  # active_norm or others
        y = rewards.cumsum()
        ax.set_ylabel("Accumulated (Normalized) Reward")
        label = "Accumulated Reward"

    ax.plot(y.index, y.values, label=label)
    ax.set_title("Accumulated Rewards")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accumulated_rewards.png", dpi=150)
    plt.close(fig)


def plot_positions(out_dir: Path, positions: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(positions.index, positions.values, label="Position")
    ax.set_title("Position Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "positions.png", dpi=150)
    plt.close(fig)


def plot_kline_with_trades(
    out_dir: Path,
    ohlcv: pd.DataFrame,
    trades: pd.DataFrame,
    lookback: int | None = 120,
    plot_start: str | None = None,
    plot_end: str | None = None,
):
    df = ohlcv.copy()

    if plot_start and plot_end:
        df = df.loc[plot_start:plot_end]
    elif lookback is not None and len(df) > lookback:
        df = df.iloc[-lookback:]

    if len(df) < 2:
        return

    tdf = trades.reindex(df.index)

    buy = pd.Series(np.nan, index=df.index)
    sell = pd.Series(np.nan, index=df.index)

    if "trade" in tdf.columns and "price" in tdf.columns:
        mb = (tdf["trade"] == "BUY").fillna(False)
        ms = (tdf["trade"] == "SELL").fillna(False)
        buy[mb] = tdf.loc[mb, "price"]
        sell[ms] = tdf.loc[ms, "price"]

    kwargs = dict(
        type="candle",
        volume=True,
        style="yahoo",
        savefig=dict(
            fname=str(out_dir / "kline_trades.png"), dpi=150, bbox_inches="tight"
        ),
    )
    ap = []
    if buy.notna().any():
        ap.append(mpf.make_addplot(buy, type="scatter", markersize=60, marker="^"))
    if sell.notna().any():
        ap.append(mpf.make_addplot(sell, type="scatter", markersize=60, marker="v"))
    if ap:
        kwargs["addplot"] = ap

    mpf.plot(df, **kwargs)


# ----------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------
def rollout(model, env: SingleAssetTradingEnv, deterministic: bool = True):
    """
    Run the trained model on the given env and collect:
        - rewards (env reward)
        - portfolio returns (from portfolio_value)
        - full actions dataframe
        - positions series
    """
    obs, info = env.reset()
    prev_pv = float(info.get("portfolio_value", 1.0))
    prev_pos = float(info.get("position", 0.0))

    rows = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        # env._t was incremented inside step(), reward corresponds to index env._t-1
        idx = env._t - 1
        idx = max(min(idx, len(env.prices) - 1), 0)
        curr_date = env.features.index[idx]

        pt = env.prices.iloc[idx]
        price_t = float(pt.iloc[0] if isinstance(pt, pd.Series) else pt)

        curr_pv = float(info.get("portfolio_value", np.nan))
        if np.isfinite(curr_pv) and prev_pv > 0:
            port_ret = curr_pv / prev_pv - 1.0
        else:
            port_ret = np.nan

        if np.isscalar(action):
            action_val = float(action)
        else:
            action_val = float(np.asarray(action).reshape(-1)[0])

        row = {
            "date": curr_date,
            "price": price_t,
            "prev_pos": prev_pos,
            "action": action_val,
            "new_pos": float(info.get("position", np.nan)),
            "turnover": float(info.get("turnover", np.nan)),
            "reward": float(reward),
            "portfolio_value": curr_pv,
            "portfolio_ret": float(port_ret),
            "trade": info.get("trade", "HOLD"),
        }
        rows.append(row)

        prev_pv = curr_pv if np.isfinite(curr_pv) else prev_pv
        prev_pos = float(info.get("position", prev_pos))

        done = bool(terminated or truncated)

    df = pd.DataFrame.from_records(rows).set_index("date")
    rewards = df["reward"].astype(float)
    port_rets = df["portfolio_ret"].fillna(0.0).astype(float)
    positions = df["new_pos"].astype(float).rename("position")
    return rewards, port_rets, df, positions


def buy_and_hold_returns(prices: pd.Series) -> pd.Series:
    """
    Daily simple returns for Buy&Hold on the asset.
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze("columns")
    ret = prices.pct_change().shift(-1).fillna(0.0)
    return ret.iloc[1:].astype(float)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="SPY")

    # data range
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)

    # env config
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--borrow-fee", type=float, default=0.0)
    parser.add_argument("--action-mode", type=str, default="target",
                        choices=["target", "delta", "discrete3"])
    parser.add_argument("--min-pos", type=float, default=-1.0)
    parser.add_argument("--max-pos", type=float, default=1.0)

    # reward / benchmark
    parser.add_argument("--reward-mode", type=str, default="active",
                        choices=["plain", "active", "active_norm"])
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Benchmark ticker for active reward; default=ticker if reward-mode!=plain")
    parser.add_argument("--bm-weight", type=float, default=1.0)
    parser.add_argument("--ema-alpha", type=float, default=0.02)

    # model + output
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "a2c", "sac", "dqn"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="artifacts/eval")
    parser.add_argument("--plot-lookback", type=int, default=120)
    parser.add_argument("--plot-start", type=str, default=None)
    parser.add_argument("--plot-end", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions at eval time")

    parser.add_argument("--advisor", type=str, default="none", choices=["none", "gru"])
    parser.add_argument("--advisor-window", type=int, default=10)
    parser.add_argument("--advisor-epochs", type=int, default=5)


    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # decide start date to download (pad some days before test-start)
    dl_start = args.start
    if args.test_start and args.start is None:
        ts = pd.Timestamp(args.test_start)
        dl_start = str((ts - BDay(args.window + 30)).date())
    if dl_start is None:
        dl_start = "1900-01-01"

    # ---------------- data & features ----------------
    raw = download_ohlcv(args.ticker, dl_start, args.end)
    raw = _flatten_columns_prioritize_fields(raw)
    ohlcv_all = _ensure_ohlcv(raw.copy())

    if "Adj Close" in raw.columns:
        price = raw["Adj Close"]
    else:
        price = raw["Close"]
    if isinstance(price, pd.DataFrame):
        price = price.squeeze("columns")

    feats = build_features(raw)

    # === NEW: apply advisor if requested ===
    if args.advisor == "gru":
        adv = fit_predict_gru(
            price,
            window=args.advisor_window,
            epochs=args.advisor_epochs,
        )
        feats = feats.join(adv.rename("adv_ret_pred")).fillna(0.0)


    # benchmark for active metrics (only needed when benchmarking)
    bm_sym = args.benchmark
    if bm_sym is None and args.reward_mode != "plain":
        bm_sym = args.ticker
    bm_ret = None
    if bm_sym is not None:
        bm_raw = download_ohlcv(bm_sym, dl_start, args.end)
        bm_raw = _flatten_columns_prioritize_fields(bm_raw)
        if "Adj Close" in bm_raw.columns:
            bm_px = bm_raw["Adj Close"]
        else:
            bm_px = bm_raw["Close"]
        if isinstance(bm_px, pd.DataFrame):
            bm_px = bm_px.squeeze("columns")
        bm_ret = bm_px.pct_change().shift(-1).fillna(0.0)

    # ---------------- choose test slice ----------------
    if args.test_start is not None:
        test_idx = raw.loc[pd.Timestamp(args.test_start):].index
    elif args.train_end is not None:
        test_idx = raw.loc[args.train_end:].index
    else:
        test_idx = raw.index

    feats_test = feats.loc[test_idx]
    price_test = price.loc[test_idx]
    ohlcv_test = ohlcv_all.loc[test_idx]
    bm_ret_test = bm_ret.loc[test_idx] if bm_ret is not None else None

    # ---------------- build env ----------------
    cfg = EnvConfig(
        window=args.window,
        transaction_cost=args.transaction_cost,
        borrow_fee_annual=args.borrow_fee,
        action_mode=args.action_mode,
        min_pos=args.min_pos,
        max_pos=args.max_pos,
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

    # ---------------- load model & rollout ----------------
    AlgoCls = ALGOS[args.algo]
    model = AlgoCls.load(args.model_path)

    rewards, port_rets, actions_df, positions = rollout(
        model, env, deterministic=not args.stochastic
    )
    rewards.to_csv(out_dir / "rewards_series.csv")

    # benchmark (Buy&Hold on the underlying)
    bh_rets = buy_and_hold_returns(price_test).reindex(rewards.index).fillna(0.0)

    # active returns (portfolio minus benchmark)
    active_rets = (port_rets - bh_rets).fillna(0.0)

    # ---------------- metrics ----------------
    def _summ(name: str, r: pd.Series):
        arr = np.asarray(r, dtype=float).reshape(-1)
        return {
            "name": name,
            "annual_return": M.annualized_return(arr),
            "annual_vol": M.annualized_vol(arr),
            "sharpe": M.sharpe_ratio(arr),
            "max_drawdown": M.max_drawdown(arr),
            "cum_return": float((1.0 + arr).prod() - 1.0),
        }

    summary_df = pd.DataFrame(
        [
            _summ("Portfolio", port_rets),
            _summ("Buy&Hold", bh_rets),
            _summ("Active", active_rets),
        ]
    ).set_index("name")

    pd.set_option("display.float_format", lambda x: f"{x:,.2%}")
    print("\n=== Evaluation (Portfolio vs Benchmark) ===")
    print(summary_df)

    # ---------------- save tables ----------------
    actions_df.to_csv(out_dir / "actions.csv")
    actions_df[actions_df["trade"] != "HOLD"].to_csv(out_dir / "trades.csv")
    print(
        f"Saved operations to: {out_dir/'actions.csv'} and {out_dir/'trades.csv'}"
    )

    # ---------------- plots ----------------
    # equity curves
    nav_port = (1.0 + port_rets).cumprod()
    nav_bh = (1.0 + bh_rets).cumprod()
    nav_active = (1.0 + active_rets).cumprod()

    plot_equity_curves(out_dir, nav_port, nav_bh)
    plot_active_equity(out_dir, nav_active)

    # accumulated reward plot (using env reward, not portfolio return)
    plot_accumulated_rewards(out_dir, rewards, args.reward_mode)

    # position over time
    plot_positions(out_dir, positions)

    # K-line with trades
    ohlcv_plot = _sanitize_ohlcv_for_mpf(ohlcv_test)
    if len(ohlcv_plot) >= 2:
        plot_kline_with_trades(
            out_dir,
            ohlcv_plot,
            actions_df,
            lookback=args.plot_lookback,
            plot_start=args.plot_start,
            plot_end=args.plot_end,
        )


if __name__ == "__main__":
    main()
