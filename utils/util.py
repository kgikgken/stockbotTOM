from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

JST = timezone(timedelta(hours=9))

def jst_now() -> datetime:
    return datetime.now(JST)

def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")

def jst_today_date():
    return jst_now().date()

def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def fmt_pct(p: float, digits: int = 1) -> str:
    if not np.isfinite(p):
        return "-"
    return f"{p*100:+.{digits}f}%"

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def slope_pct(series: pd.Series, lookback: int = 5) -> float:
    if series is None or len(series) < lookback + 1:
        return float("nan")
    a = float(series.iloc[-1])
    b = float(series.iloc[-1 - lookback])
    if not (np.isfinite(a) and np.isfinite(b) and b != 0):
        return float("nan")
    return (a / b) - 1.0

def last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return float("nan")
