# ============================================
# utils/features.py
# 指標計算（ATR/RSI/MA/リターン等）
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class Features:
    close: float
    open: float
    high: float
    low: float
    volume: float

    sma20: float
    sma50: float
    sma10: float

    rsi14: float
    atr14: float
    atrp14: float  # ATR%

    ret5d: float
    ret20d: float

    gu_flag: bool


def compute_features(df: pd.DataFrame) -> Optional[Features]:
    if df is None or df.empty or len(df) < 60:
        return None

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            return None

    d = df.dropna().copy()
    if len(d) < 60:
        return None

    close = d["Close"]
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma10 = _sma(close, 10)
    rsi14 = _rsi(close, 14)
    atr14 = _atr(d, 14)

    c = float(d["Close"].iloc[-1])
    o = float(d["Open"].iloc[-1])
    h = float(d["High"].iloc[-1])
    l = float(d["Low"].iloc[-1])
    v = float(d["Volume"].iloc[-1])

    s20 = float(sma20.iloc[-1])
    s50 = float(sma50.iloc[-1])
    s10 = float(sma10.iloc[-1])
    rsi = float(rsi14.iloc[-1])
    atr = float(atr14.iloc[-1]) if not np.isnan(float(atr14.iloc[-1])) else 0.0
    atrp = (atr / c * 100.0) if c > 0 else 0.0

    # 5d/20d return (%)
    ret5 = 0.0
    ret20 = 0.0
    if len(close) >= 6:
        ret5 = float((c / float(close.iloc[-6]) - 1.0) * 100.0)
    if len(close) >= 21:
        ret20 = float((c / float(close.iloc[-21]) - 1.0) * 100.0)

    # GU判定: Open > PrevClose + 1.0 ATR
    prev_close = float(close.iloc[-2])
    gu_flag = False
    if atr > 0 and o > (prev_close + 1.0 * atr):
        gu_flag = True

    return Features(
        close=c,
        open=o,
        high=h,
        low=l,
        volume=v,
        sma20=s20,
        sma50=s50,
        sma10=s10,
        rsi14=rsi,
        atr14=atr,
        atrp14=atrp,
        ret5d=ret5,
        ret20d=ret20,
        gu_flag=gu_flag,
    )