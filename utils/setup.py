from __future__ import annotations

import numpy as np
import pandas as pd

def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan

def sma(series: pd.Series, w: int) -> float:
    if series is None or len(series) < w:
        return _last(series)
    return float(series.rolling(w).mean().iloc[-1])

def rsi14(close: pd.Series) -> float:
    if close is None or len(close) < 20:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return float((100 - (100 / (1 + rs))).iloc[-1])

def atr14(df: pd.DataFrame) -> float:
    if df is None or len(df) < 20:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = tr.rolling(14).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan

def sma_slope_up(series: pd.Series, w: int = 20, lookback: int = 5) -> bool:
    if series is None or len(series) < w + lookback + 2:
        return False
    ma = series.rolling(w).mean()
    a = float(ma.iloc[-1])
    b = float(ma.iloc[-(lookback + 1)])
    if not (np.isfinite(a) and np.isfinite(b) and b != 0):
        return False
    return (a / b - 1.0) > 0.0

def hh20(close: pd.Series) -> float:
    if close is None or len(close) < 25:
        return np.nan
    return float(close.rolling(20).max().iloc[-2])
