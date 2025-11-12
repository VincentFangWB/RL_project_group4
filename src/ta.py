from __future__ import annotations
import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Classic Wilder-style RSI with simple rolling means (educational baseline).
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line




def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Build feature DataFrame aligned with df.index.
    # Includes: log_return, pct_change, SMA10/20, RSI14, MACD, MACD_signal, 20D volatility.
    features = pd.DataFrame(index=df.index)
    # robust price extraction as a Series
    if "Adj Close" in df.columns:
        close = df["Adj Close"]
    elif "Close" in df.columns:
        close = df["Close"]
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in DataFrame columns.")

    # yfinance or prior ops can yield duplicate labels -> a single-label selection may be a DataFrame.
    if isinstance(close, pd.DataFrame):
        # if multiple columns share the same label, take the first; or choose a policy you prefer
        close = close.iloc[:, 0]

    close = close.astype(float)
    close.name = "close"  # set Series name instead of calling .rename("close")

    features["log_ret"] = np.log(close).diff().fillna(0.0)
    features["ret"] = close.pct_change().fillna(0.0)
    features["sma10"] = close.rolling(10).mean()
    features["sma20"] = close.rolling(20).mean()
    features["rsi14"] = rsi(close, 14)
    macd_line, sig = macd(close)
    features["macd"] = macd_line
    features["macd_sig"] = sig
    features["vol20"] = features["log_ret"].rolling(20).std().fillna(0.0)

    features = features.bfill().fillna(0.0)
    return features.astype(np.float32)