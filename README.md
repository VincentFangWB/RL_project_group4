# RL Project Group 4 – Deep RL for Portfolio Management

This repository contains the code for a course project on **deep reinforcement learning (RL) for trading and portfolio management**.  

We go from **single-asset SPY trading** to **multi-asset portfolio allocation**, and finally to a **cash-aware portfolio with different market regimes** (uptrend / range-bound / downtrend).

The main focus is on comparing:

- Classic policy-gradient RL (PPO, A2C)
- Value-based RL (DQN)
- An **enhanced DQN** with dueling architecture + Double Q-learning + prioritized replay + template-based portfolio actions

across different asset universes and market conditions.

> **Data Period Used in Experiments**  
> Unless otherwise specified, all experiments in this project use:  
> - **Training period:** 2010-01-01 to 2022-12-31  
> - **Evaluation period:** 2023-01-01 to 2025-11-12  
>  
> The example commands below are written to match this split.

---

## 1. Repository Structure

At the top level you will see:

```text
RL_project_group4/
├── model/          # Saved models (trained PPO / DQN / enhanced DQN weights)
├── result/         # Backtest results: equity curves, CSV logs, metrics tables, comparison plots
└── src/            # All source code
```

Inside `src/` the structure is:

```text
src/
├── data/
│   ├── single_asset_loader.py      # Load SPY or other single-asset OHLCV with yfinance
│   └── multi_asset_loader.py       # Load price matrices for multiple tickers, build return matrices
│
├── envs/
│   ├── single_asset_env.py         # Gymnasium env for single-asset continuous position [-1, 1]
│   ├── multi_asset_portfolio_env.py# Multi-asset env without explicit cash (weights sum to 1 across stocks)
│   ├── multi_asset_template_env.py # Multi-asset env with discrete “template” actions (PPO-free / DQN baseline)
│   └── multi_asset_template_cash_env.py
│       # New: multi-asset + explicit cash dimension, with discrete templates (keep, growth tilt, defensive tilt, high cash)
│
├── models/
│   ├── train_single_ppo.py         # Train PPO agent on single asset (e.g., SPY)
│   ├── train_single_a2c.py         # (if used) Train A2C agent on single asset
│   ├── train_single_dqn.py         # Train DQN on single asset (discrete position changes)
│   ├── train_multi_ppo.py          # Train PPO on multi-asset portfolio (no cash)
│   ├── train_multi_a2c.py          # Train A2C on multi-asset portfolio (no cash)
│   ├── train_multi_dqn.py          # Train vanilla DQN on multi-asset (no cash)
│   ├── train_enhanced_dqn.py       # Enhanced DQN for multi-asset portfolio (no cash)
│   └── train_enhanced_dqn_template_cash.py
│       # Enhanced DQN with discrete templates + explicit cash (uptrend / range / downtrend experiment)
│
├── backtest/
│   ├── evaluate_single_asset.py               # Evaluate PPO / DQN / baseline on SPY
│   ├── evaluate_multi_asset.py                # Evaluate multi-asset algorithms vs equal-weight & S&P 500
│   ├── evaluate_enhanced_dqn.py               # Evaluate enhanced DQN (no cash) vs PPO / A2C / vanilla DQN
│   ├── evaluate_enhanced_dqn_template_cash.py # Evaluate enhanced DQN (template + cash) in three regimes
│   └── plot_multi_algos_nav.py                # Combine NAV curves from multiple algorithms into one figure
│
└── utils/
    ├── metrics.py                  # Annualized return, volatility, Sharpe, max drawdown, etc.
    └── plotting.py                 # Helpers for equity curves and diagnostic plots
```

> **Note:** The exact filenames may evolve slightly. If you rename any script in the repo, just update the README commands accordingly.

---

## 2. Environment Setup & How to Run

### 2.1. Python environment

You can use `venv`, `conda`, or `mamba`. Example with `venv`:

```bash
git clone https://github.com/VincentFangWB/RL_project_group4.git
cd RL_project_group4

# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

### 2.2. Example 1 – Train & Evaluate PPO on Single Asset (SPY)

In this example we strictly follow the global data split:

- **Training:** 2010-01-01 to 2022-12-31  
- **Evaluation:** 2023-01-01 to 2025-11-12  

#### Train PPO on SPY (2010-01-01 to 2022-12-31)

```bash
python -m src.models.train_single_ppo   --ticker SPY   --start 2010-01-01   --end   2022-12-31   --train-end 2022-12-31   --window 30   --transaction-cost 0.001   --total-timesteps 300000   --save-dir model/ppo_spy   --tensorboard-log tb_logs/ppo_spy
```

Typical flags:

- `--ticker`: symbol to trade (SPY)  
- `--start` / `--end`: full dataset range used for training  
- `--train-end`: last date used for training (here equal to `--end`)  
- `--window`: lookback window (days) for state features  
- `--transaction-cost`: proportional cost per unit turnover  
- `--total-timesteps`: number of RL environment steps  
- `--save-dir`: where to store `model.zip` and logs  

#### Evaluate PPO on SPY (2023-01-01 to 2025-11-12)

```bash
python -m src.backtest.evaluate_single_asset   --ticker SPY   --start 2023-01-01   --end   2025-11-12   --window 30   --algo ppo   --model-path model/ppo_spy/model.zip   --transaction-cost 0.001   --out-dir result/eval_spy_ppo
```

This will typically write into `result/eval_spy_ppo/`:

- `equity_curve.png` – NAV of PPO vs SPY buy-and-hold  
- `actions.csv` – daily position / action / reward  
- `metrics_table.txt` – cumulative return, annualized return, volatility, Sharpe, max drawdown  

You can rerun the evaluation with `--algo dqn` and `--model-path` pointing to your DQN model to compare.

---

### 2.3. Example 2 – Train & Evaluate Multi-Asset PPO (No Cash, Uptrend Basket)

Here we use 10 “mostly uptrend” large caps, with the same time split:

- **Training:** 2010-01-01 to 2022-12-31  
- **Evaluation:** 2023-01-01 to 2025-11-12  

The uptrend basket:

> `NVDA, AAPL, TSLA, GS, BAC, XOM, WMT, KO, UNH, MCD`

#### Train PPO on multi-asset portfolio (2010-01-01 to 2022-12-31)

```bash
python -m src.models.train_multi_ppo   --tickers "NVDA,AAPL,TSLA,GS,BAC,XOM,WMT,KO,UNH,MCD"   --start 2010-01-01   --end   2022-12-31   --window 30   --transaction-cost 0.001   --risk-lambda 0.1   --rolling-window 30   --max-steps 2000   --total-timesteps 300000   --n-envs 4   --learning-rate 3e-4   --batch-size 2048   --gamma 0.99   --ent-coef 0.0   --clip-range 0.2   --save-path model/ppo_multi_up/model.zip   --tensorboard-log tb_logs/ppo_multi_up
```

#### Evaluate multi-asset PPO vs equal-weight & S&P 500 (2023-01-01 to 2025-11-12)

```bash
python -m src.backtest.evaluate_multi_asset   --algo ppo   --model-path model/ppo_multi_up/model.zip   --tickers "NVDA,AAPL,TSLA,GS,BAC,XOM,WMT,KO,UNH,MCD"   --start 2023-01-01   --end   2025-11-12   --window 30   --transaction-cost 0.001   --risk-lambda 0.1   --rolling-window 30   --max-steps 2000   --out-dir result/eval_multi_ppo_up
```

Outputs:

- RL NAV vs **equal-weight portfolio** (1/N across the 10 stocks)  
- NAV vs **S&P 500** (ticker `^GSPC`)  
- Metrics table for all three  

You can run the same evaluation with `--algo dqn` or by calling 
`evaluate_enhanced_dqn.py` to compare PPO / A2C / vanilla DQN / **enhanced DQN**.

---

### 2.4. Example 3 – Train & Evaluate Enhanced DQN with Cash (Three Market Regimes)

For the “cash-aware” experiment, we use a **template-based discrete action env** with cash and keep the same global time split:

- **Training:** 2010-01-01 to 2022-12-31  
- **Evaluation:** 2023-01-01 to 2025-11-12  

**Baskets**

- **Uptrend basket (10 stocks)**:  
  `NVDA,AAPL,TSLA,GS,BAC,XOM,WMT,KO,UNH,MCD`
- **Range-bound basket (3 stocks)**:  
  `INTC,F,AA`
- **Downtrend basket (2 stocks)**:  
  `VEON,PRGO`

#### Train enhanced DQN (template + cash, 2010-01-01 to 2022-12-31)

```bash
# Uptrend
python -m src.models.train_enhanced_dqn_template_cash   --tickers "NVDA,AAPL,TSLA,GS,BAC,XOM,WMT,KO,UNH,MCD"   --start 2010-01-01   --end   2022-12-31   --total-timesteps 1000000   --save-path model/enhanced_dqn_cash_up.pt

# Range
python -m src.models.train_enhanced_dqn_template_cash   --tickers "INTC,F,AA"   --start 2010-01-01   --end   2022-12-31   --total-timesteps 1000000   --save-path model/enhanced_dqn_cash_range.pt

# Downtrend
python -m src.models.train_enhanced_dqn_template_cash   --tickers "VEON,PRGO"   --start 2010-01-01   --end   2022-12-31   --total-timesteps 1000000   --save-path model/enhanced_dqn_cash_down.pt
```

Each action chooses one of a small set of **portfolio templates**:

- keep current weights  
- equal-weight all stocks (0% cash)  
- growth tilt vs defensive tilt  
- high-cash template (e.g., 70% cash, 30% risky assets)  

This is designed to test whether the DQN can **“hide in cash”** during downtrends.

#### Evaluate enhanced DQN vs equal-weight & S&P 500 (2023-01-01 to 2025-11-12)

```bash
# Example: downtrend basket
python -m src.backtest.evaluate_enhanced_dqn_template_cash   --model-path model/enhanced_dqn_cash_down.pt   --tickers "VEON,PRGO"   --start 2023-01-01   --end   2025-11-12   --window 30   --transaction-cost 0.001   --risk-lambda 0.1   --rolling-window 30   --max-steps 2000   --out-dir result/eval_enhanced_dqn_cash_down
```

Outputs include:

- `equity_curve_enhanced_dqn_template_cash.png`  
- `actions_enhanced_dqn_template_cash.csv` (with `w_cash` column)  
- `metrics_table_enhanced_dqn_template_cash.txt`  

---

## 3. Experiments & Findings

This repo supports three main experiments, all using the same time split
(**train: 2010-01-01 to 2022-12-31; eval: 2023-01-01 to 2025-11-12**):

1. **Single-Asset (SPY) – PPO / A2C / DQN**  
2. **Multi-Asset Uptrend (10 stocks, no cash) – PPO / A2C / DQN / Enhanced DQN**  
3. **Multi-Asset with Cash, Three Regimes – Enhanced DQN (uptrend / range / downtrend)**  

Below is a high-level summary plus reasoning.

---

### 3.1. Experiment 1 – Single-Asset SPY (PPO / A2C / DQN vs Buy & Hold)

**Setup**

- Asset: **SPY** (S&P 500 ETF)  
- Algorithms: PPO, A2C, DQN  
- Environment: single-asset continuous position in `[-1, 1]` (short to long), daily rebalancing  
- Training period: **2010-01-01 to 2022-12-31**  
- Evaluation period: **2023-01-01 to 2025-11-12**  
- Baseline: SPY **buy-and-hold**  

**Empirical pattern (qualitative)**

- **Buy-and-hold SPY** is extremely hard to beat out-of-sample.  
- PPO / A2C sometimes match or slightly outperform in some in-sample periods, but
  tend to **track SPY** with slightly different risk.  
- Vanilla DQN often underperforms due to:
  - more volatile policies,
  - over-trading, and
  - instability of value-based methods in low-signal, long-horizon tasks.

**Reasons**

1. **Single asset + strong long-term uptrend ⇒ optimal policy ≈ buy & hold.**  
   There is no diversification: the only decisions are timing leverage / shorting.  
   In long SPY bull markets, most “smart” timing rules actually hurt performance after
   transaction costs.

2. **Efficient Market Hypothesis (EMH).**  
   EMH suggests it is hard to consistently beat broad index ETFs like SPY with timing alone, because prices quickly incorporate available information.  

3. **RL algorithms are not designed for ultra-low signal-to-noise, long-horizon tasks.**  
   Many DRL portfolio papers find that:
   - RL can outperform benchmarks in specific backtests or markets,  
   - but **out-of-sample robustness is weak**, and results are sensitive to hyper-parameters and time windows.  

4. **Transaction costs + exploration.**  
   PPO / A2C / DQN explore by changing positions; small noisy forecasts plus costs mean
   that “doing something” is often worse than “doing nothing” (buy-and-hold).

**Takeaway:**  
For a single highly efficient index like SPY, **buy-and-hold is incredibly strong**. RL mostly ends up learning a noisy approximation to buy-and-hold, and often underperforms once you account for costs.

---

### 3.2. Experiment 2 – Multi-Asset Uptrend, No Cash  
**(NVDA,AAPL,TSLA,GS,BAC,XOM,WMT,KO,UNH,MCD)**

**Setup**

- 10 large caps with strong long-term performance.  
- Algorithms: **PPO, A2C, vanilla DQN, enhanced DQN (Double-Dueling + PER + templates)**.  
- No explicit cash: the agent always stays fully invested across the 10 stocks.  
- Training period: **2010-01-01 to 2022-12-31**  
- Evaluation period: **2023-01-01 to 2025-11-12**  
- Baselines:
  - **Equal-weight (1/N) portfolio** across the 10 names.  
  - **S&P 500** as an external benchmark.

**Empirical pattern (qualitative)**

- Equal-weight across these 10 strong performers is already a **very strong baseline**.  
- PPO & A2C:
  - sometimes lean toward growth names (NVDA, TSLA, AAPL),
  - but can be noisy and sensitive to hyper-parameters.  
- Vanilla DQN:
  - tends to over-react and sometimes over-concentrate,  
  - can be hurt by instability and function approximation error.  
- **Enhanced DQN** (with template actions and risk-aware reward):
  - often shows **better risk-adjusted performance** within this set,
  - more stable NAV path than vanilla DQN,
  - sometimes modestly outperforms PPO / A2C in this very specific uptrend basket.

**Why can enhanced DQN look better here?**

1. **Reduced action complexity via templates.**  
   Instead of directly outputting arbitrary weight vectors, enhanced DQN chooses from a **small library of portfolio templates** (growth tilt, defensive tilt, etc.).  
   This:
   - lowers the effective action dimensionality,
   - makes Q-learning more stable,
   - allows the agent to focus on deciding **which “style rotation” to apply** rather than micro-tuning each asset weight.

2. **Dueling architecture + Double Q-learning + prioritized replay.**  
   These tricks improve value estimation and reduce over-optimistic Q-values, which is crucial in noisy financial data.  

3. **Uptrend universe is forgiving.**  
   Because the basket is biased toward long-term winners, many reasonable allocation schemes (growth tilt, sector tilt, etc.) can look good ex-post. A well-tuned RL model can “ride winners” and avoid a few laggards, producing higher Sharpe in such favorable conditions.  

4. **Equal-weight is still hard to beat.**  
   The “1/N puzzle” literature shows that equal-weight portfolios are surprisingly competitive due to robustness to estimation error.  
   Enhanced DQN’s apparent outperformance in this single chosen period / universe should be viewed as **conditional** and not guaranteed in general.

---

### 3.3. Experiment 3 – Three Market Regimes with Cash  
**(Enhanced DQN, template actions + cash)**

**Baskets**

- **Uptrend (10 stocks)** – same as above:  
  `NVDA,AAPL,TSLA,GS,BAC,XOM,WMT,KO,UNH,MCD`  
- **Range-bound (3 stocks)**:  
  `INTC,F,AA`  
- **Downtrend (3 stocks)**:  
  `VEON,PRGO,M`  

**Environment**

- There is now a **cash dimension**:
  - Weights over `N` stocks + 1 cash = 1.  
  - One of the discrete templates is **“high cash”** (e.g., 70% cash, 30% risky assets).  
- Enhanced DQN chooses among 5 templates:
  - keep current weights  
  - equal-weight all stocks  
  - growth tilt  
  - defensive tilt  
  - high-cash  

**Empirical pattern (qualitative)**

Across **all three regimes (train: 2010-01-01 to 2022-12-31, eval: 2023-01-01 to 2025-11-12)**, enhanced DQN (template + cash) **does not consistently beat the equal-weight benchmark**:

- **Uptrend basket (with cash):**
  - RL sometimes holds too much cash during strong uptrends,
  - or rotates between styles too often, incurring transaction costs.
  - Equal-weight fully invested in strong names remains very competitive.

- **Range-bound basket:**
  - Market has no clear trend; price noise dominates.
  - Any attempt to time short-term moves tends to generate churn and costs.
  - Equal-weight with low turnover is again robust.

- **Downtrend basket:**
  - Ideally, DQN **should** learn to stay near the high-cash template most of the time.
  - In practice, it often:
    - over-reacts to short bear-market rallies,
    - oscillates between “risk-on” and “risk-off” templates,
    - and never perfectly learns “just stay in cash and do nothing.”
  - As a result, its total loss may be smaller than full-equity, but **still worse than (or not clearly better than) equal-weight baseline**.

**Why does enhanced DQN still struggle vs equal-weight in this regime experiment?**

1. **Signal-to-noise ratio is very low.**  
   Financial return predictability is weak; RL is trying to infer a complex policy from noisy price changes. Many DRL portfolio studies find that RL agents often overfit to specific regimes and fail to generalize to new conditions.  

2. **Equal-weight is extremely robust to estimation error.**  
   1/N diversification performs surprisingly well when expected returns and covariances are difficult to estimate. Even sophisticated mean-variance or ML-based portfolios can struggle to beat it out-of-sample.  

3. **RL objective mismatch and horizon mismatch.**  
   - The training reward is local (daily or weekly) and risk-adjusted (return minus volatility penalty).  
   - The *evaluation* is global (multi-year Sharpe / max drawdown).  
   RL may exploit short-term patterns that do not translate into superior long-term Sharpe.

4. **RL loves “doing something,” but the optimal action is often “do nothing.”**  
   In downtrends, the best policy is often “stay in cash for a very long time.”  
   - This is difficult for exploration-based RL: exploration inherently means taking trades.  
   - “Doing nothing for months” gets little immediate reward signal; the agent may over-respond to short-term rallies.

5. **Limited training data & non-stationarity.**  
   There are only a few truly long, sustained downtrend periods in the data. RL may under-sample such regimes during training, then fail to generalize to new bear markets.  

**Takeaway:**  
Even with explicit cash and a structurally “safe” high-cash template, enhanced DQN **does not reliably learn to stay in cash** during downtrends, nor does it consistently beat equal-weight across regimes. Equal-weight remains a tough baseline.

---

## 4. Overall Conclusions & Lessons Learned

Based on the experiments in this repo and the broader literature:

### 4.1. Beating S&P 500 (SPY) is Very Hard

- SPY tracks the S&P 500 – a highly diversified, highly liquid, and relatively efficient market.  
- EMH suggests that **systematically beating such indices is extremely difficult**, especially net of transaction costs and risk.  
- DRL portfolio papers sometimes show benchmark-beating results in specific backtests, but:
  - often only over particular periods,
  - with significant hyper-parameter sensitivity,
  - and limited robustness across markets and regimes.  

In our SPY experiment, buy-and-hold is a very strong baseline that RL rarely outperforms out-of-sample (2010-01-01 to 2022-12-31 training, tested on 2023-01-01 to 2025-11-12).

---

### 4.2. RL in Investing – Key Limitations

1. **Sample inefficiency & non-stationarity.**  
   RL typically requires huge amounts of experience; financial markets provide limited historical data with changing regimes. Models trained on one regime can break in another.  

2. **Low signal-to-noise environments.**  
   Price changes are noisy; true predictive signals are tiny. RL can easily overfit to noise, especially with high-capacity neural networks.

3. **Objective mismatch.**  
   RL optimizes a step-wise reward (e.g., daily return minus risk penalty), but investors care about **long-horizon Sharpe, drawdowns, tail risk, liquidity, etc.**  
   It is non-trivial to encode all of this into a simple scalar reward.  

4. **Transaction costs & market impact.**  
   Frequent rebalancing magnifies costs. RL often learns high-turnover policies unless heavily penalized; real-world market impact is even worse than backtests assume.

5. **Fragility & hyper-parameter sensitivity.**  
   Recent benchmark studies show that DRL portfolio methods are sensitive to random seeds, tuning, and network architecture, and can exhibit large variance in outcomes.  

---

### 4.3. Why Equal-Weight Often Wins (or at Least Competes)

The **1/N (equal-weight) portfolio** is sometimes called “naive,” but:

- It is robust to estimation error in expected returns and covariances.  
- Empirical studies show that 1/N often performs competitively with, or even better than, more complex parametric portfolios when inputs are noisy.  

Our experiments reproduce this intuition: equal-weight across carefully chosen stocks is **very hard for RL to beat consistently**, even when using a long training period (2010-01-01 to 2022-12-31) and a relatively long evaluation window (2023-01-01 to 2025-11-12).

---

### 4.4. How Could We Improve This Line of Work?

Some directions suggested by both your experiments and recent research:

1. **Use DRL as a layer on top of robust portfolio building blocks.**  
   - Instead of directly choosing weights for hundreds of assets, RL can choose among a **small set of pre-built diversified model portfolios** (e.g., minimum-variance, risk-parity, factor portfolios).  
   - This is similar in spirit to frameworks that combine MPT with RL, such as Deep Portfolio Optimization.

2. **Better state representations & regime awareness.**  
   - Include volatility regimes, macro indicators, market breadth, credit spreads, etc.  
   - Embed market conditions to better balance risk and return.  

3. **Stronger risk-aware rewards and constraints.**  
   - Penalize turnover more aggressively.  
   - Explicitly penalize drawdowns or CVaR, not just variance.  
   - Use Sharpe or Omega-ratio inspired rewards with smoothing.  

4. **Ensembles & model averaging.**  
   - Ensemble multiple RL agents (PPO, SAC, DQN, etc.) to reduce variance.  

5. **Walk-forward and online learning evaluation.**  
   - Use strict walk-forward splits and rolling re-training to mimic real-time deployment.  
   - Evaluate robustness under small changes in hyper-parameters, start dates, and universes.  

---

## 5. References

Below are some representative papers and resources related to DQN and deep reinforcement learning in asset management and trading:

1. **Mnih, V. et al. (2015).** *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529–533.  
   - The original DQN paper that introduced deep Q-learning for high-dimensional control tasks.

2. **Deng, Y., Bao, F., & Kong, Y. (2016).** *Deep Direct Reinforcement Learning for Financial Signal Representation and Trading.*  
   - Applies deep reinforcement learning with a DQN-like architecture to financial trading.

3. **Jiang, Z., Xu, D., & Liang, J. (2017).** *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.*  
   - Proposes a deep RL framework for dynamic portfolio allocation.

4. **Zhang, Z., Zohren, S., & Roberts, S. (2020).** *Deep Reinforcement Learning for Trading.*  
   - Surveys and benchmarks DRL methods (including policy-gradient and value-based) on financial markets.

5. **AWS Machine Learning Blog.** *Automated decision-making with deep reinforcement learning.*  
   - https://aws.amazon.com/blogs/machine-learning/automated-decision-making-with-deep-reinforcement-learning/  
   - Describes how to use deep RL on AWS for end-to-end automated decision-making, including financial use cases.
