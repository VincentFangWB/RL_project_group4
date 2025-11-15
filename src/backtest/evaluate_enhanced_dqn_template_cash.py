# src/backtest/evaluate_enhanced_dqn_template_cash.py

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data.multi_asset_loader import load_prices_matrix
from src.envs.multi_asset_template_cash_env import (
    MultiAssetTemplateCashConfig,
    MultiAssetTemplateCashEnv,
)
from src.models.train_enhanced_dqn_template_cash import DuelingQNetwork


def compute_metrics(nav: pd.Series, trading_days_per_year: int = 252) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return dict(
            cumulative_return=0.0,
            annual_return=0.0,
            annual_vol=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
        )
    ret = nav.pct_change().dropna()
    cum_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    n = len(ret)
    ann_ret = float((1.0 + cum_ret) ** (trading_days_per_year / n) - 1.0)
    ann_vol = float(ret.std() * np.sqrt(trading_days_per_year))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    max_dd = float(dd.min())

    return dict(
        cumulative_return=cum_ret,
        annual_return=ann_ret,
        annual_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )


def format_table(metrics_rl: dict, metrics_eqw: dict, metrics_sp500: dict) -> str:
    lines = []
    lines.append("=== Enhanced DQN (template + cash) Metrics ===")
    for k in ["cumulative_return", "annual_return", "annual_vol", "sharpe", "max_drawdown"]:
        lines.append(f"{k:20s}: {metrics_rl.get(k, 0.0): .6f}")
    lines.append("")
    lines.append("=== Equal-Weight Benchmark Metrics ===")
    for k in ["cumulative_return", "annual_return", "annual_vol", "sharpe", "max_drawdown"]:
        lines.append(f"{k:20s}: {metrics_eqw.get(k, 0.0): .6f}")
    lines.append("")
    lines.append("=== S&P 500 Benchmark Metrics ===")
    for k in ["cumulative_return", "annual_return", "annual_vol", "sharpe", "max_drawdown"]:
        lines.append(f"{k:20s}: {metrics_sp500.get(k, 0.0): .6f}")
    return "\n".join(lines)


def build_eqw_and_sp500(
    tickers: List[str],
    start: str,
    end: str,
) -> Tuple[pd.Series, pd.Series]:
    prices, dates = load_prices_matrix(tickers=tickers, start=start, end=end)
    num_assets = len(tickers)
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    asset_ret = price_df.pct_change().fillna(0.0)
    w_eq = np.ones(num_assets, dtype=np.float32) / float(num_assets)
    port_ret = asset_ret.values @ w_eq
    nav_eqw = (1.0 + pd.Series(port_ret, index=asset_ret.index)).cumprod()

    sp500_prices, sp500_dates = load_prices_matrix(["^GSPC"], start=start, end=end)
    sp500_series = pd.Series(sp500_prices[:, 0], index=sp500_dates)
    sp_ret = sp500_series.pct_change().fillna(0.0)
    nav_sp500 = (1.0 + sp_ret).cumprod()
    return nav_eqw, nav_sp500


def run_eval(
    model_path: str,
    tickers: List[str],
    start: str,
    end: str,
    window: int,
    transaction_cost: float,
    risk_lambda: float,
    rolling_window: int,
    max_steps: int,
    out_dir: str,
    device: str = "cpu",
):
    os.makedirs(out_dir, exist_ok=True)

    prices, dates = load_prices_matrix(tickers=tickers, start=start, end=end)
    cfg = MultiAssetTemplateCashConfig(
        tickers=tickers,
        start=start,
        end=end,
        window=window,
        transaction_cost=transaction_cost,
        risk_lambda=risk_lambda,
        rolling_window=rolling_window,
        max_steps=max_steps,
    )
    env = MultiAssetTemplateCashEnv(config=cfg, prices=prices, dates=dates)

    nav_eqw_full, nav_sp500_full = build_eqw_and_sp500(tickers, start, end)

    obs_space = env.observation_space
    act_space = env.action_space
    state_dim = obs_space.shape[0]
    action_dim = act_space.n

    device_t = torch.device(device)
    q_net = DuelingQNetwork(state_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path, map_location=device_t))
    q_net.to(device_t)
    q_net.eval()

    obs, info = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    done = False
    records = []

    while not done:
        s = torch.from_numpy(obs).unsqueeze(0).to(device_t)
        with torch.no_grad():
            q_vals = q_net(s)
            action = int(torch.argmax(q_vals, dim=1).item())

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = terminated or truncated

        current_date = dates[env.current_idx]
        w = step_info["weights"]
        rec = {
            "date": current_date,
            "step": env.current_step,
            "nav": step_info["nav"],
            "reward": float(reward),
            "portfolio_return": step_info["portfolio_return"],
            "portfolio_return_net": step_info["portfolio_return_net"],
            "turnover": step_info["turnover"],
            "cost": step_info["cost"],
        }
        for i, t in enumerate(tickers):
            rec[f"w_{t}"] = float(w[i])
        rec["w_cash"] = float(w[-1])
        records.append(rec)

        obs = next_obs

    actions_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    nav_rl = pd.Series(actions_df["nav"].values, index=actions_df["date"])
    start_date = nav_rl.index[0]
    end_date = nav_rl.index[-1]

    nav_eqw = nav_eqw_full.loc[start_date:end_date]
    nav_sp500 = nav_sp500_full.loc[start_date:end_date]

    metrics_rl = compute_metrics(nav_rl)
    metrics_eqw = compute_metrics(nav_eqw)
    metrics_sp500 = compute_metrics(nav_sp500)

    table_str = format_table(metrics_rl, metrics_eqw, metrics_sp500)
    print(table_str)
    with open(os.path.join(out_dir, "metrics_table_enhanced_dqn_template_cash.txt"), "w") as f:
        f.write(table_str)

    actions_df.to_csv(os.path.join(out_dir, "actions_enhanced_dqn_template_cash.csv"), index=False)

    # Plot equity curves (normalized to 1 at start)
    fig, ax = plt.subplots(figsize=(10, 5))
    nav_rl_norm = nav_rl / nav_rl.iloc[0]
    nav_eqw_norm = nav_eqw / nav_eqw.iloc[0]
    nav_sp500_norm = nav_sp500 / nav_sp500.iloc[0]
    ax.plot(nav_rl_norm.index, nav_rl_norm.values, label="Enhanced DQN (template+cash)")
    ax.plot(nav_eqw_norm.index, nav_eqw_norm.values, label="Equal-Weight", linestyle="--")
    ax.plot(nav_sp500_norm.index, nav_sp500_norm.values, label="S&P 500", linestyle=":")

    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized NAV")
    ax.set_title("Enhanced DQN (template+cash) vs Equal-Weight & S&P 500")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equity_curve_enhanced_dqn_template_cash.png"), dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Enhanced DQN (template+cash) on multi-asset env."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated list of tickers.",
    )
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--risk-lambda", type=float, default=0.1)
    parser.add_argument("--rolling-window", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run_eval(
        model_path=args.model_path,
        tickers=tickers,
        start=args.start,
        end=args.end,
        window=args.window,
        transaction_cost=args.transaction_cost,
        risk_lambda=args.risk_lambda,
        rolling_window=args.rolling_window,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
