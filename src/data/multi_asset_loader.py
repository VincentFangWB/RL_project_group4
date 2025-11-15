# src/data/multi_asset_loader.py

from typing import List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf


def load_prices_matrix(
    tickers: List[str],
    start: str,
    end: str,
    column: str = "Adj Close",
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Download price data for multiple tickers from yfinance and return a price matrix.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols, e.g. ["NVDA", "AAPL", "TSLA", ...].
    start : str
        Start date in "YYYY-MM-DD".
    end : str
        End date in "YYYY-MM-DD".
    column : str
        Preferred column of the yfinance output to use, e.g. "Adj Close" or "Close".

    Returns
    -------
    prices : np.ndarray
        Price matrix of shape [T, N] (time, assets) with dtype float32.
    index : pd.DatetimeIndex
        Date index corresponding to the rows of the price matrix.
    """
    # Explicitly set auto_adjust=False so that "Adj Close" is always present
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )

    # Defensive check: no data
    if data is None or len(data) == 0:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    # Multi-asset case: yfinance returns a MultiIndex (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # If the preferred column does not exist (e.g. "Adj Close"), fall back to "Close"
        top_level_names = list(data.columns.get_level_values(0).unique())
        effective_column = column if column in top_level_names else "Close"

        prices_df = data[effective_column]
        # Ensure column order matches the tickers list
        prices_df = prices_df[tickers]

    else:
        # Single-asset case: columns are simple Index
        if column in data.columns:
            prices_df = data[[column]]
        elif "Close" in data.columns:
            prices_df = data[["Close"]]
        else:
            raise KeyError(
                f"Neither '{column}' nor 'Close' found in downloaded data columns: {list(data.columns)}"
            )

    prices_df = prices_df.dropna()
    prices = prices_df.values.astype(np.float32)
    return prices, prices_df.index
