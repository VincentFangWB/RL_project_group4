# src/backtest/plot_multi_algos_nav.py

import argparse
import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from src.data.multi_asset_loader import load_prices_matrix


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_nav_series(csv_path: str, label: str) -> pd.Series:
    """
    Load NAV time series from an actions CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the actions CSV file.
    label : str
        Label for this series (algorithm name).

    Returns
    -------
    nav_series : pd.Series
        NAV series indexed by datetime.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{label}: CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "nav" not in df.columns:
        raise ValueError(f"{label}: CSV must contain 'date' and 'nav' columns.")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    nav_series = df["nav"].astype(float)
    return nav_series


def build_equal_weight_and_sp500(
    tickers: List[str],
    start: str,
    end: str,
) -> (pd.Series, pd.Series):
    """
    Build equal-weight portfolio NAV and S&P 500 NAV over [start, end].
    """
    # Multi-asset prices
    prices, dates = load_prices_matrix(tickers=tickers, start=start, end=end)
    num_assets = len(tickers)

    # IMPORTANT: set columns=tickers so columns align with asset names
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)

    # Daily returns
    asset_ret = price_df.pct_change().fillna(0.0)  # [T, N]

    # Equal-weight weights as a numpy array
    w_eq = np.ones(num_assets, dtype=np.float32) / float(num_assets)

    # Portfolio return = matrix multiply of returns and weights
    port_ret = asset_ret.values @ w_eq  # shape [T]

    # Equal-weight NAV (start from 1.0)
    nav_eqw = (1.0 + pd.Series(port_ret, index=asset_ret.index)).cumprod()

    # S&P 500 prices
    sp500_prices, sp500_dates = load_prices_matrix(
        tickers=["^GSPC"],
        start=start,
        end=end,
    )
    sp500_series = pd.Series(sp500_prices[:, 0], index=sp500_dates)
    sp500_ret = sp500_series.pct_change().fillna(0.0)
    nav_sp500 = (1.0 + sp500_ret).cumprod()

    return nav_eqw, nav_sp500



def plot_multi_algos_nav(
    series_dict: Dict[str, pd.Series],
    nav_eqw: pd.Series,
    nav_sp500: pd.Series,
    out_path: str,
):
    """
    Plot normalized NAV curves for multiple RL algorithms plus
    equal-weight and S&P 500 on the same figure.

    All series are normalized to start at 1.0 on their own first date.
    """
    if not series_dict:
        raise ValueError("No RL NAV series provided to plot.")

    plt.figure(figsize=(10, 5))

    # Plot RL algorithms
    for label, nav in series_dict.items():
        if nav.empty:
            continue
        nav_norm = nav / nav.iloc[0]
        plt.plot(nav.index, nav_norm.values, label=label)

    # Equal-weight benchmark
    if not nav_eqw.empty:
        nav_eqw_norm = nav_eqw / nav_eqw.iloc[0]
        plt.plot(nav_eqw_norm.index, nav_eqw_norm.values,
                 label="Equal-Weight", linestyle="--")

    # S&P 500 benchmark
    if not nav_sp500.empty:
        nav_sp500_norm = nav_sp500 / nav_sp500.iloc[0]
        plt.plot(nav_sp500_norm.index, nav_sp500_norm.values,
                 label="S&P 500", linestyle=":")

    plt.xlabel("Date")
    plt.ylabel("Normalized NAV")
    plt.title("Multi-Asset NAV: PPO / A2C / DQN / Enhanced DQN vs Benchmarks")
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined NAV figure to: {out_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot multi-asset NAV curves for PPO / A2C / DQN / Enhanced DQN "
                    "plus Equal-Weight and S&P 500."
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default="result",
        help="Root directory containing eval_multi_ppo, eval_multi_a2c, eval_multi_dqn, eval_enhanced_dqn.",
    )
    parser.add_argument(
        "--ppo-subdir",
        type=str,
        default="eval_multi_ppo",
        help="Subdirectory for PPO evaluation results (contains actions.csv).",
    )
    parser.add_argument(
        "--a2c-subdir",
        type=str,
        default="eval_multi_a2c",
        help="Subdirectory for A2C evaluation results (contains actions.csv).",
    )
    parser.add_argument(
        "--dqn-subdir",
        type=str,
        default="eval_multi_dqn",
        help="Subdirectory for vanilla DQN evaluation results (contains actions.csv).",
    )
    parser.add_argument(
        "--enh-dqn-subdir",
        type=str,
        default="eval_enhanced_dqn",
        help="Subdirectory for enhanced DQN evaluation results (contains actions_enhanced_dqn.csv).",
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default="NVDA,AAPL,TSLA,GS,BAC,XOM,WMT,KO,UNH,MCD",
        help="Comma-separated list of tickers used in the multi-asset environment.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-02-16",
        help="Start date (YYYY-MM-DD) for equal-weight and S&P 500 benchmarks.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-11-11",
        help="End date (YYYY-MM-DD) for equal-weight and S&P 500 benchmarks.",
    )

    parser.add_argument(
        "--out-path",
        type=str,
        default="result/multi_algos_equity_curve_with_benchmarks.png",
        help="Output path for the combined NAV figure.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    series_dict: Dict[str, pd.Series] = {}

    # PPO
    ppo_csv = os.path.join(args.root_dir, args.ppo_subdir, "actions.csv")
    try:
        series_dict["PPO"] = load_nav_series(ppo_csv, "PPO")
    except Exception as e:
        print(f"[WARN] Skipping PPO: {e}")

    # A2C
    a2c_csv = os.path.join(args.root_dir, args.a2c_subdir, "actions.csv")
    try:
        series_dict["A2C"] = load_nav_series(a2c_csv, "A2C")
    except Exception as e:
        print(f"[WARN] Skipping A2C: {e}")

    # Vanilla DQN
    dqn_csv = os.path.join(args.root_dir, args.dqn_subdir, "actions.csv")
    try:
        series_dict["DQN"] = load_nav_series(dqn_csv, "DQN")
    except Exception as e:
        print(f"[WARN] Skipping DQN: {e}")

    # Enhanced DQN
    enh_dqn_csv = os.path.join(args.root_dir, args.enh_dqn_subdir, "actions_enhanced_dqn.csv")
    try:
        series_dict["Enhanced DQN"] = load_nav_series(enh_dqn_csv, "Enhanced DQN")
    except Exception as e:
        print(f"[WARN] Skipping Enhanced DQN: {e}")

    if not series_dict:
        print("No RL NAV series loaded. Please check your paths.")
        return

    # Build equal-weight and S&P 500 NAV
    nav_eqw, nav_sp500 = build_equal_weight_and_sp500(
        tickers=tickers,
        start=args.start,
        end=args.end,
    )

    plot_multi_algos_nav(series_dict, nav_eqw, nav_sp500, args.out_path)


if __name__ == "__main__":
    main()
