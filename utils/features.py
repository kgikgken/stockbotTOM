from __future__ import annotations

import numpy as np
import pandas as pd


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def sma(series: pd.Series, window: int) -> float:
    if series is None or len(series) < window:
        return _last(series)
    return float(series.rolling(window).mean().iloc[-1])


def rsi(close: pd.Series, period: int = 14) -> float:
    if close is None or len(close) < period + 2:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    out = 100 - (100 / (1 + rs))
    return float(out.iloc[-1]) if np.isfinite(out.iloc[-1]) else np.nan


def turnover_avg(df: pd.DataFrame, window: int = 20) -> float:
    if df is None or len(df) < window:
        return np.nan
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
    t = (close * vol).rolling(window).mean().iloc[-1]
    return float(t) if np.isfinite(t) else np.nan