from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_EPS = 1e-12


def _series_1d(x) -> pd.Series:
    """
    Coerce inputs like list/np.ndarray/Series/DataFrame to a 1D float Series.
    - Squeezes (N,1) or (1,N) to (N,)
    - If still not 1D, flattens in row-major order (ravel)
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.squeeze("columns")
        else:
            arr = np.asarray(x).ravel()
            return pd.Series(arr, dtype=float)

    arr = np.asarray(x).squeeze()
    if arr.ndim != 1:
        arr = arr.ravel()
    return pd.Series(arr, dtype=float)


def cumulative_returns(returns) -> pd.Series:
    r = _series_1d(returns)
    return (1.0 + r).cumprod() - 1.0


def annualized_return(returns, freq: int = TRADING_DAYS) -> float:
    r = _series_1d(returns)
    n = len(r)
    if n == 0:
        return 0.0
    total_ret = float((1.0 + r).prod() - 1.0)
    years = n / float(max(freq, 1))
    if years <= 0:
        return 0.0
    return (1.0 + total_ret) ** (1.0 / years) - 1.0


def annualized_vol(returns, freq: int = TRADING_DAYS) -> float:
    r = _series_1d(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(freq))


def sharpe_ratio(returns, rf: float = 0.0, freq: int = TRADING_DAYS) -> float:
    r = _series_1d(returns)
    if len(r) < 2:
        return 0.0
    per_period_rf = rf / float(freq)
    excess = r - per_period_rf
    vol = float(excess.std(ddof=1) * np.sqrt(freq))
    if vol <= _EPS:
        return 0.0
    mean_excess_ann = float(excess.mean() * freq)
    return mean_excess_ann / vol


def max_drawdown(returns) -> float:
    r = _series_1d(returns)
    if len(r) == 0:
        return 0.0
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())