from __future__ import annotations
import pandas as pd
import yfinance as yf

def download_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    # Download OHLCV data via yfinance and return a clean DataFrame indexed by date.
    # Columns: [Open, High, Low, Close, Adj Close, Volume]
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if df.empty:
        raise ValueError(f"No data returned for {ticker} {start}~{end}")
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df