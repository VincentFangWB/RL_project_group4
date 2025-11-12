# RL_project_group4

This project provides a clean, **reproducible** starting point to apply RL to trading:
- Fetch historical market data with **`yfinance`**
- A **Gymnasium** custom environment for **single‑asset, continuous position** control in `[-1, 1]`
- **Training scripts** for **PPO**, **A2C**, and **SAC** (Stable‑Baselines3)
- **Evaluation** vs. **Buy & Hold** with metrics (Annualized Return, Volatility, Sharpe, Max Drawdown)
- **NEW**: Evaluation now exports
  - Net‑asset‑value (NAV) / equity curve plot
  - Full operations table (per‑day action, buy/sell, turnover)
  - Candlestick (K‑line) plot with **buy/sell markers**
  - Optional position‑over‑time plot
- Unit tests + GitHub Actions CI

## Repository Layout
```
rl-trading/
├── README.md
├── pyproject.toml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── features/
│   │   └── ta.py
│   ├── envs/
│   │   └── single_asset_env.py
│   ├── models/
│   │   ├── train_ppo.py
│   │   ├── train_a2c.py
│   │   └── train_sac.py
│   ├── backtest/
│   │   └── evaluate.py
│   └── utils/
│       └── metrics.py
└── tests/
    ├── test_env.py
    └── test_metrics.py
```

## Quickstart

### 1) Create environment & install
```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .[dev]
```

### 2) Train (choose one)
```bash
# PPO
python -m src.models.train_ppo \
  --ticker SPY \
  --start 2009-01-01 --end 2024-12-31 \
  --train-end 2022-12-31 \
  --window 30 \
  --total-timesteps 200000 \
  --transaction-cost 0.001 \
  --save-dir artifacts/ppo_spy

# A2C
python -m src.models.train_a2c \
  --ticker SPY --start 2009-01-01 --end 2024-12-31 --train-end 2022-12-31 \
  --window 30 --total-timesteps 200000 --transaction-cost 0.001 \
  --save-dir artifacts/a2c_spy

# SAC
python -m src.models.train_sac \
  --ticker SPY --start 2009-01-01 --end 2024-12-31 --train-end 2022-12-31 \
  --window 30 --total-timesteps 200000 --transaction-cost 0.001 \
  --save-dir artifacts/sac_spy
```

### 3) Evaluate (now with plots & exports)
```bash
python -m src.backtest.evaluate \
  --ticker SPY \
  --start 2024-10-15 \
  --end 2025-11-12 \
  --train-end 2024-12-31 \
  --window 30 \
  --algo ppo \
  --model-path artifacts/ppo_spy/model.zip \
  --out-dir artifacts/eval_spy \
  --plot-lookback 120  # last 120 trading days for K-line plot (optional)

python -m src.backtest.compare \
  --ticker SPY \
  --start 2024-10-15 \
  --end 2025-11-12 \
  --train-end 2024-12-31 \
  --window 30 \
  --transaction-cost 0.001 \
  --out-dir artifacts/compare_spy \
  --algos ppo a2c sac \
  --model ppo=artifacts/ppo_spy/model.zip \
  --model a2c=artifacts/a2c_spy/model.zip \
  --model sac=artifacts/sac_spy/model.zip \
  --plot-lookback 120

```
This writes to `--out-dir`:
- `equity_curve.png` (NAV curves: RL vs Buy&Hold)
- `positions.png` (position time series)
- `kline_trades.png` (candlestick with buy/sell markers)
- `actions.csv` (all operations, day by day)
- `trades.csv` (only buy/sell rows)
- Console shows a metrics table (annualized return/vol, Sharpe, max drawdown, cumulative return)