# src/data/loader.py
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
import yfinance as yf


# ---------- helpers: canonical field mapping ----------
def _canon_field(name: str) -> str | None:
    k = str(name).strip().lower().replace("_", " ").replace("-", " ")
    if "adj" in k and "close" in k:  # prioritize Adj Close if present
        return "Adj Close"
    if "open" in k:   return "Open"
    if "high" in k:   return "High"
    if "low" in k:    return "Low"
    if "close" in k:  return "Close"
    if "volume" in k: return "Volume"
    return None


def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten/normalize columns to standard OHLCV names, robust to yfinance MultiIndex.
    Handles both (Price, Ticker) and (Ticker, Price) and single-level with prefixes.
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # decide which level contains field names by counting matches
        scores = []
        for lvl in range(out.columns.nlevels):
            vals = out.columns.get_level_values(lvl)
            score = sum(_canon_field(v) is not None for v in vals)
            scores.append(score)
        field_level = int(np.argmax(scores))  # level most likely with field names

        new_cols: list[str] = []
        for col in out.columns.to_flat_index():
            parts = list(col) if isinstance(col, tuple) else [col]
            field = _canon_field(parts[field_level])
            # fallback: try the "other" level if this one didn't decode
            if field is None and len(parts) > 1:
                other = 1 - field_level
                field = _canon_field(parts[other])
            new_cols.append(field if field else str(parts[field_level]))
        out.columns = new_cols
    else:
        out.columns = [(_canon_field(c) or str(c)) for c in out.columns]

    # Ensure a usable Close
    if "Close" not in out.columns:
        if "Adj Close" in out.columns:
            out["Close"] = out["Adj Close"]
        elif "Open" in out.columns:
            out["Close"] = out["Open"]
        else:
            # If you ever see repeated tickers like ['SPY','SPY',...], it means
            # columns were corrupted upstream. Using group_by='column' avoids this. :contentReference[oaicite:2]{index=2}
            raise ValueError(f"Cannot locate OHLCV fields in columns: {list(df.columns)}")

    # Keep OHLCV if present; leave others (e.g., Dividends) out
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
    return out[keep].copy()


def _to_numeric_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    return out


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV with stable column schema for downstream code.
    Notes:
      - yfinance 'end' is EXCLUSIVE; 'start' is inclusive. :contentReference[oaicite:3]{index=3}
      - group_by='column' prevents multi-ticker MultiIndex in most cases. :contentReference[oaicite:4]{index=4}
      - If a MultiIndex still occurs (library updates), we flatten it.
    """
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,   # keep both Close & Adj Close if Yahoo provides it
        actions=False,       # no Dividends/Splits columns
        group_by="column",   # prefer single-level columns
        progress=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned by yfinance for {ticker} {start}..{end}")

    # Some yfinance versions still emit MultiIndex (library changes). Flatten robustly. :contentReference[oaicite:5]{index=5}
    df = _flatten_ohlcv(df)
    df = _to_numeric_ohlcv(df)
    return df
