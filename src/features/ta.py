# src/features/ta.py
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- helpers: normalize yfinance columns ----------
def _clean_key(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def _flatten_columns_prioritize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map any MultiIndex or prefixed single-level columns to canonical names:
    Open, High, Low, Close, Adj Close, Volume. Leaves other cols intact.
    Works for both (Ticker, Field) and (Field, Ticker).
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        new_cols = []
        for col in out.columns:
            parts = [str(x) for x in col]
            field = None
            # Prefer Adj Close if present in any level
            for p in parts:
                k = _clean_key(p)
                if "adj" in k and "close" in k:
                    field = "Adj Close"
                    break
            if field is None:
                for p in parts:
                    k = _clean_key(p)
                    if k in ("open", "high", "low", "close", "volume"):
                        field = {"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume"}[k]
                        break
            new_cols.append(field if field else str(col[-1]))
        out.columns = new_cols
    else:
        new_cols = []
        for c in out.columns:
            k = _clean_key(c)
            if "adj" in k and "close" in k:
                new_cols.append("Adj Close")
            elif "open" in k:
                new_cols.append("Open")
            elif "high" in k:
                new_cols.append("High")
            elif "low" in k:
                new_cols.append("Low")
            elif "close" in k:
                new_cols.append("Close")
            elif "volume" in k:
                new_cols.append("Volume")
            else:
                new_cols.append(str(c))
        out.columns = new_cols

    return out


def _ensure_close_exists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a 'Close' series exists; fall back to 'Adj Close' then 'Open'.
    """
    out = df.copy()
    if "Close" not in out.columns:
        if "Adj Close" in out.columns:
            out["Close"] = out["Adj Close"]
        elif "Open" in out.columns:
            out["Close"] = out["Open"]
        else:
            raise ValueError(f"Cannot find Close/Adj Close/Open in columns: {list(out.columns)}")
    return out


def _to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------- main feature builder ----------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of features aligned to the input index with at least:
      - ret: next-day simple return (what the env expects)
      - rsi14, z20, roc10: simple technicals

    The function is robust to yfinance's MultiIndex or prefixed columns and
    guarantees there is a numeric 'Close' (fallback to 'Adj Close'/'Open').

    Note: yfinance 'end' is EXCLUSIVE; 'start' is inclusive. Download accordingly.  # FYI
    """
    x = _flatten_columns_prioritize_fields(df.copy())
    x = _ensure_close_exists(x)

    # prefer Adj Close if present, else Close
    close = x["Adj Close"] if "Adj Close" in x.columns else x["Close"]
    close = pd.to_numeric(close, errors="coerce")

    # --- core label: next-day return for RL reward alignment
    ret = close.pct_change().shift(-1).fillna(0.0).rename("ret")

    # --- a few light, stable features ---
    # RSI(14) (EWMA variant; robust and differentiable-ish)
    win = 14
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / win, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / win, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi14 = (100 - 100 / (1 + rs)).fillna(50.0).rename("rsi14")

    # 20-day z-score
    ma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std()
    z20 = ((close - ma20) / (std20.replace(0, np.nan) + 1e-12)).fillna(0.0).rename("z20")

    # 10-day rate of change
    roc10 = close.pct_change(10).fillna(0.0).rename("roc10")

    feats = pd.concat([ret, rsi14, z20, roc10], axis=1)
    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feats.index = pd.to_datetime(feats.index, errors="coerce")
    feats = feats[~feats.index.isna()]

    return feats
