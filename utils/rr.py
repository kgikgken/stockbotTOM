from __future__ import annotations

import numpy as np
import pandas as pd

def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan

def atr14(df: pd.DataFrame) -> float:
    if df is None or len(df) < 16:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    v = tr.rolling(14).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan

def hh20(df: pd.DataFrame) -> float:
    close = df["Close"].astype(float)
    if len(close) < 21:
        return float(close.max())
    return float(close.tail(20).max())

def swing_low(df: pd.DataFrame, lookback: int = 12) -> float:
    low = df["Low"].astype(float)
    if len(low) < lookback:
        return float(low.min())
    return float(low.tail(lookback).min())

def build_trade_plan(hist: pd.DataFrame, setup_type: str, in_center: float, in_low: float, in_high: float, mkt_score: int) -> dict:
    df = hist.copy()
    close = df["Close"].astype(float)
    price = _last(close)
    atr = atr14(df)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(price * 0.01, 1.0)

    if setup_type == "A":
        stop = in_low - 0.7 * atr
        stop = min(stop, swing_low(df, 12) - 0.2 * atr)
    else:
        stop = in_center - 1.0 * atr
        stop = min(stop, swing_low(df, 12) - 0.2 * atr)

    risk = max(in_center - stop, 0.5 * atr)
    stop = in_center - risk

    tp1 = in_center + 1.5 * risk
    tp2 = in_center + 3.0 * risk

    if mkt_score >= 70:
        tp2 *= 1.02
        tp1 *= 1.01
    elif mkt_score <= 45:
        tp2 *= 0.98
        tp1 *= 0.99

    rr = (tp2 - in_center) / (in_center - stop) if in_center > stop else 0.0

    expected_days = (tp2 - in_center) / (1.0 * atr) if atr > 0 else 999.0
    expected_days = float(np.clip(expected_days, 0.5, 12.0))
    r_per_day = rr / expected_days if expected_days > 0 else 0.0

    return {
        "atr": float(atr),
        "in_center": float(in_center),
        "in_low": float(in_low),
        "in_high": float(in_high),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "expected_days": float(expected_days),
        "r_per_day": float(r_per_day),
        "price_now": float(price) if np.isfinite(price) else np.nan,
    }
