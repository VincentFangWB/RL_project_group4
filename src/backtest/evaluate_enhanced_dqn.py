# src/backtest/evaluate_enhanced_dqn.py

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data.multi_asset_loader import load_prices_matrix
from src.envs.multi_asset_portfolio_env import (
    MultiAssetConfig,
    MultiAssetPortfolioTemplateDQNEnv,
)
from src.models.train_multi_enhanced_dqn import DuelingQNetwork


DEFAULT_TICKERS: List[str] = [
    "NVDA", "AAPL", "TSLA", "GS", "BAC",
    "XOM", "WMT", "KO", "UNH", "MCD",
]


# ============================================================
# Metrics & helper functions
# ============================================================

def compute_metrics(nav_series: pd.Series, trading_days_per_year: int = 252) -> dict:
    """
    Compute basic performance metrics from a NAV time series.

    Parameters
    ----------
    nav_series : pd.Series
        Portfolio NAV indexed by date (or step).
    trading_days_per_year : int
        Number of trading days used for annualization.

    Returns
    -------
    metrics : dict
        Dictionary of performance metrics:
        - cumulative_return
        - annual_return
        - annual_vol
        - sharpe
        - max_drawdown
    """
    nav_series = nav_series.dropna()
    if len(nav_series) < 2:
        return {
            "cumulative_return": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    returns = nav_series.pct_change().dropna()
    cumulative_return = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1.0)

    n = len(returns)
    annual_return = float((1.0 + cumulative_return) ** (trading_days_per_year / n) - 1.0)
    annual_vol = float(returns.std() * np.sqrt(trading_days_per_year))

    if annual_vol > 0:
        sharpe = float(annual_return / annual_vol)
    else:
        sharpe = 0.0

    running_max = nav_series.cummax()
    drawdown = nav_series / running_max - 1.0
    max_drawdown = float(drawdown.min())

    metrics = {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
    return metrics


def build_equal_weight_nav(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    num_assets: int,
    initial_nav: float = 1.0,
) -> pd.Series:
    """
    Build an equal-weight buy-and-hold NAV series for the same asset universe.

    Parameters
    ----------
    prices : np.ndarray
        Price matrix of shape [T, N].
    dates : pd.DatetimeIndex
        Date index of length T.
    num_assets : int
        Number of assets N.
    initial_nav : float
        Initial NAV for the equal-weight portfolio.

    Returns
    -------
    nav_eqw : pd.Series
        Equal-weight portfolio NAV indexed by dates.
    """
    price_df = pd.DataFrame(prices, index=dates)
    asset_returns = price_df.pct_change().fillna(0.0)  # [T, N]
    w_eq = np.ones(num_assets, dtype=np.float32) / float(num_assets)
    port_ret = asset_returns.values @ w_eq  # [T]
    nav = initial_nav * (1.0 + pd.Series(port_ret, index=dates)).cumprod()
    return nav


def format_metrics_table(
    metrics_rl: dict,
    metrics_eqw: Optional[dict] = None,
    metrics_sp500: Optional[dict] = None,
) -> str:
    """
    Format metrics into a text table similar to:

    === RL Portfolio Metrics ===
    cumulative_return   :  0.960466
    annual_return       :  0.280552
    ...

    Returns
    -------
    table_str : str
        Multi-line string containing formatted metrics.
    """
    lines: List[str] = []

    # RL metrics
    lines.append("=== Enhanced DQN Portfolio Metrics ===")
    for k in ["cumulative_return", "annual_return", "annual_vol", "sharpe", "max_drawdown"]:
        v = metrics_rl.get(k, 0.0)
        lines.append(f"{k:20s}: {v: .6f}")
    lines.append("")  # blank line

    # Equal-weight metrics (optional)
    if metrics_eqw is not None:
        lines.append("=== Equal-Weight Benchmark Metrics ===")
        for k in ["cumulative_return", "annual_return", "annual_vol", "sharpe", "max_drawdown"]:
            v = metrics_eqw.get(k, 0.0)
            lines.append(f"{k:20s}: {v: .6f}")
        lines.append("")

    # S&P 500 metrics (optional)
    if metrics_sp500 is not None:
        lines.append("=== S&P 500 Benchmark Metrics ===")
        for k in ["cumulative_return", "annual_return", "annual_vol", "sharpe", "max_drawdown"]:
            v = metrics_sp500.get(k, 0.0)
            lines.append(f"{k:20s}: {v: .6f}")
        lines.append("")

    table_str = "\n".join(lines)
    return table_str


# ============================================================
# Plotting helpers
# ============================================================

def plot_equity_curve(
    actions_df: pd.DataFrame,
    nav_eqw: pd.Series,
    nav_sp500: pd.Series,
    out_path: str,
):
    """
    Plot equity curve (NAV) for the enhanced DQN portfolio, equal-weight
    benchmark, and S&P 500 buy-and-hold benchmark.

    All series are normalized to start at 1.0 on the first evaluation date,
    so the curves share the same starting point.
    """
    if actions_df.empty:
        return

    dates = actions_df["date"]

    # RL portfolio NAV (normalize to 1.0 at the first evaluation date)
    nav_rl = actions_df["nav"].values.astype(float)
    nav_rl_norm = nav_rl / nav_rl[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(dates, nav_rl_norm, label="Enhanced DQN NAV")

    # Equal-weight benchmark
    if not nav_eqw.empty:
        nav_eqw_aligned = nav_eqw.reindex(dates).ffill().astype(float).values
        nav_eqw_norm = nav_eqw_aligned / nav_eqw_aligned[0]
        ax.plot(
            dates,
            nav_eqw_norm,
            label="Equal-Weight NAV",
            linestyle="--",
        )

    # S&P 500 benchmark
    if not nav_sp500.empty:
        nav_sp500_aligned = nav_sp500.reindex(dates).ffill().astype(float).values
        nav_sp500_norm = nav_sp500_aligned / nav_sp500_aligned[0]
        ax.plot(
            dates,
            nav_sp500_norm,
            label="S&P 500 NAV",
            linestyle=":",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized NAV")
    ax.set_title("Equity Curve (Enhanced DQN vs Benchmarks, normalized)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_positions(actions_df: pd.DataFrame, tickers: List[str], out_path: str):
    """
    Plot time series of portfolio weights for each asset and cash.
    """
    if actions_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for ticker in tickers:
        col = f"w_{ticker}"
        if col in actions_df.columns:
            ax.plot(actions_df["date"], actions_df[col], label=col)

    if "w_cash" in actions_df.columns:
        ax.plot(actions_df["date"], actions_df["w_cash"], label="w_cash")

    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weights Over Time (Enhanced DQN)")
    ax.legend(loc="upper right", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rewards(actions_df: pd.DataFrame, out_path: str):
    """
    Plot reward per step over time.
    """
    if actions_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(actions_df["date"], actions_df["reward"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Reward")
    ax.set_title("Step Reward Over Time (Enhanced DQN)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_actions_and_trades(actions_df: pd.DataFrame, out_dir: str):
    """
    Save actions.csv and trades.csv to the output directory.

    actions.csv:
        All steps with NAV, reward, weights, etc.

    trades.csv:
        Only rows where there was a non-zero turnover (i.e., rebalancing).
    """
    actions_path = os.path.join(out_dir, "actions_enhanced_dqn.csv")
    actions_df.to_csv(actions_path, index=False)

    trades_df = actions_df[actions_df["turnover"].abs() > 1e-6].copy()
    trades_path = os.path.join(out_dir, "trades_enhanced_dqn.csv")
    trades_df.to_csv(trades_path, index=False)


# ============================================================
# Core evaluation: rollout enhanced DQN policy once
# ============================================================

def run_evaluation_enhanced_dqn(
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
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Run a single evaluation episode for the enhanced DQN (Double-Dueling + PER)
    model on the multi-asset template-action environment.

    Parameters
    ----------
    model_path : str
        Path to the saved PyTorch model (.pt file).
    tickers : List[str]
        List of tickers used by the environment.
    start : str
        Start date for price data.
    end : str
        End date for price data.
    window : int
        Historical window length for the environment.
    transaction_cost : float
        Transaction cost coefficient.
    risk_lambda : float
        Risk penalty coefficient.
    rolling_window : int
        Rolling window for risk estimation.
    max_steps : int
        Maximum number of steps per episode.
    out_dir : str
        Output directory (created if it does not exist).
    device : str
        "cpu" or "cuda".

    Returns
    -------
    actions_df : pd.DataFrame
        DataFrame containing step-by-step actions and diagnostics.
    nav_eqw_rl_range : pd.Series
        Equal-weight NAV series aligned with the RL evaluation range.
    nav_sp500_rl_range : pd.Series
        S&P 500 buy-and-hold NAV series aligned with the RL evaluation range.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load price matrix and date index for the multi-asset universe.
    prices, date_index = load_prices_matrix(tickers=tickers, start=start, end=end)
    num_assets = len(tickers)

    # Equal-weight benchmark over full period
    nav_eqw_full = build_equal_weight_nav(
        prices=prices,
        dates=date_index,
        num_assets=num_assets,
        initial_nav=1.0,
    )

    # S&P 500 benchmark (^GSPC) over full period
    sp500_ticker = ["^GSPC"]
    sp500_prices, sp500_dates = load_prices_matrix(
        tickers=sp500_ticker,
        start=start,
        end=end,
    )
    sp500_series = pd.Series(sp500_prices[:, 0], index=sp500_dates)
    sp500_returns = sp500_series.pct_change().fillna(0.0)
    nav_sp500_full = 1.0 * (1.0 + sp500_returns).cumprod()

    # Build environment config and env
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

    # Build Q-network and load weights
    obs_space = env.observation_space
    act_space = env.action_space
    assert len(obs_space.shape) == 1, "Enhanced DQN expects flat observation vectors."

    state_dim = obs_space.shape[0]
    action_dim = act_space.n

    device_t = torch.device(device)
    q_net = DuelingQNetwork(state_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path, map_location=device_t))
    q_net.to(device_t)
    q_net.eval()

    # Roll out a single episode with greedy policy (no epsilon exploration).
    obs, info = env.reset()
    obs = np.asarray(obs, dtype=np.float32)

    done = False
    records = []

    while not done:
        s_tensor = torch.from_numpy(obs).unsqueeze(0).to(device_t)
        with torch.no_grad():
            q_values = q_net(s_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)

        current_date = date_index[env.current_idx]

        weights = step_info["weights"]
        record = {
            "date": current_date,
            "step": env.current_step,
            "nav": step_info["nav"],
            "reward": float(reward),
            "portfolio_return": step_info["portfolio_return"],
            "portfolio_return_net": step_info["portfolio_return_net"],
            "turnover": step_info["turnover"],
            "cost": step_info["cost"],
        }

        for i, ticker in enumerate(tickers):
            record[f"w_{ticker}"] = float(weights[i])
        record["w_cash"] = float(weights[-1])

        records.append(record)

        obs = next_obs
        done = terminated or truncated

        if done:
            break

    actions_df = pd.DataFrame(records)
    actions_df.sort_values("date", inplace=True)
    actions_df.reset_index(drop=True, inplace=True)

    # Align benchmarks to the RL evaluation date range.
    if not actions_df.empty:
        start_date = actions_df["date"].iloc[0]
        end_date = actions_df["date"].iloc[-1]

        nav_eqw_rl_range = nav_eqw_full.loc[start_date:end_date]
        nav_sp500_rl_range = nav_sp500_full.loc[start_date:end_date]
    else:
        nav_eqw_rl_range = pd.Series(dtype=float)
        nav_sp500_rl_range = pd.Series(dtype=float)

    return actions_df, nav_eqw_rl_range, nav_sp500_rl_range


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate enhanced DQN (Double-Dueling + PER) on multi-asset portfolio env."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the enhanced DQN model (.pt file).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated list of tickers.",
    )
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2025-11-10")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--risk-lambda", type=float, default=0.1)
    parser.add_argument("--rolling-window", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=2000)

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for plots and CSV files.",
    )
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    actions_df, nav_eqw, nav_sp500 = run_evaluation_enhanced_dqn(
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

    if actions_df.empty:
        print("No actions recorded during evaluation (empty episode).")
        return

    # Metrics for RL
    nav_series = pd.Series(actions_df["nav"].values, index=actions_df["date"])
    metrics_rl = compute_metrics(nav_series)

    # Metrics for equal-weight benchmark
    metrics_eqw = None
    if not nav_eqw.empty:
        nav_eqw_aligned = nav_eqw.reindex(actions_df["date"]).ffill()
        metrics_eqw = compute_metrics(nav_eqw_aligned)

    # Metrics for S&P 500
    metrics_sp500 = None
    if not nav_sp500.empty:
        nav_sp500_aligned = nav_sp500.reindex(actions_df["date"]).ffill()
        metrics_sp500 = compute_metrics(nav_sp500_aligned)

    # Format and print metrics table
    metrics_table = format_metrics_table(
        metrics_rl=metrics_rl,
        metrics_eqw=metrics_eqw,
        metrics_sp500=metrics_sp500,
    )
    print(metrics_table)

    metrics_path = os.path.join(args.out_dir, "metrics_table_enhanced_dqn.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_table)

    # Save actions & trades
    save_actions_and_trades(actions_df, args.out_dir)

    # Plots
    equity_path = os.path.join(args.out_dir, "equity_curve_enhanced_dqn.png")
    plot_equity_curve(actions_df, nav_eqw, nav_sp500, equity_path)

    positions_path = os.path.join(args.out_dir, "positions_enhanced_dqn.png")
    plot_positions(actions_df, tickers, positions_path)

    rewards_path = os.path.join(args.out_dir, "reward_curve_enhanced_dqn.png")
    plot_rewards(actions_df, rewards_path)

    print(f"\nSaved outputs to: {args.out_dir}")
    print("  - equity_curve_enhanced_dqn.png")
    print("  - positions_enhanced_dqn.png")
    print("  - reward_curve_enhanced_dqn.png")
    print("  - actions_enhanced_dqn.csv")
    print("  - trades_enhanced_dqn.csv")
    print("  - metrics_table_enhanced_dqn.txt")


if __name__ == "__main__":
    main()
