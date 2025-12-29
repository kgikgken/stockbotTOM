　from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float("nan")


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.rolling(n).mean() / (dn.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class Feat:
    close: float
    prev_close: float
    open_: float
    high: float
    low: float
    volume: float

    ma20: float
    ma50: float
    ma10: float
    ma20_slope_5d: float

    rsi14: float
    atr14: float
    atr_pct: float
    hh20: float
    vol_ma20: float
    turnover_ma20: float  # JPY proxy（Close*Vol）

    off_ma20_atr: float   # (Close - MA20)/ATR


def compute_features(hist: pd.DataFrame) -> Optional[Feat]:
    if hist is None or len(hist) < 80:
        return None

    df = hist.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float) if "Open" in df.columns else close
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    ma10 = sma(close, 10)
    ma20_slope_5d = ma20.pct_change(5)

    rsi14 = rsi(close, 14)
    atr14 = atr(df, 14)

    hh20 = close.rolling(20).max()
    vol_ma20 = vol.rolling(20).mean()
    turnover_ma20 = (vol * close).rolling(20).mean()

    c = _last(close)
    pc = float(close.iloc[-2]) if len(close) >= 2 else c
    o = _last(open_)
    h = _last(high)
    l = _last(low)
    v = _last(vol)

    ma20v = _last(ma20)
    ma50v = _last(ma50)
    ma10v = _last(ma10)
    slope = _last(ma20_slope_5d)

    r = _last(rsi14)
    a = _last(atr14)
    a = a if np.isfinite(a) and a > 0 else max(c * 0.01, 1.0)

    atr_pct = float(a / c) if np.isfinite(c) and c > 0 else float("nan")

    hh20v = _last(hh20)
    vm20 = _last(vol_ma20)
    tm20 = _last(turnover_ma20)

    off = float((c - ma20v) / a) if np.isfinite(ma20v) and np.isfinite(c) else float("nan")

    return Feat(
        close=float(c),
        prev_close=float(pc),
        open_=float(o),
        high=float(h),
        low=float(l),
        volume=float(v),
        ma20=float(ma20v),
        ma50=float(ma50v),
        ma10=float(ma10v),
        ma20_slope_5d=float(slope),
        rsi14=float(r),
        atr14=float(a),
        atr_pct=float(atr_pct),
        hh20=float(hh20v),
        vol_ma20=float(vm20),
        turnover_ma20=float(tm20),
        off_ma20_atr=float(off),
    )


def rs_20d(stock_close: pd.Series, index_close: pd.Series) -> float:
    """相対強度：20日リターン差（%）。"""
    try:
        if stock_close is None or index_close is None:
            return float("nan")
        if len(stock_close) < 21 or len(index_close) < 21:
            return float("nan")
        s = stock_close.astype(float)
        i = index_close.astype(float)
        r_s = (s.iloc[-1] / s.iloc[-21] - 1.0) * 100.0
        r_i = (i.iloc[-1] / i.iloc[-21] - 1.0) * 100.0
        return float(r_s - r_i)
    except Exception:
        return float("nan")