from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

@dataclass
class FeaturePack:
    close: float
    open: float
    high: float
    low: float
    prev_close: float
    sma10: float
    sma20: float
    sma50: float
    rsi14: float
    atr14: float
    atr_pct: float
    hh20: float
    vol: float
    vol_ma20: float
    turnover: float
    adv20: float
    sma20_slope_5d: float
    rs_20d: float  # relative strength vs index (placeholder; can be 0 if not provided)

def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float("nan")

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if len(rsi) else float("nan")

def calc_features(hist: pd.DataFrame, index_hist: Optional[pd.DataFrame] = None) -> FeaturePack:
    df = hist.copy()
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    c = _last(close)
    o = _last(open_)
    h = _last(high)
    l = _last(low)
    pc = float(close.iloc[-2]) if len(close) >= 2 else c

    sma10 = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else c
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else c
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else c

    rsi14 = _rsi(close, 14)
    atr14 = _atr(df, 14)
    if not np.isfinite(atr14) or atr14 <= 0:
        atr14 = max(c * 0.01, 1.0)

    atr_pct = float(atr14 / c * 100.0) if np.isfinite(c) and c > 0 else float("nan")

    hh20 = float(high.rolling(20).max().iloc[-1]) if len(high) >= 20 else float(_last(high))

    v = _last(vol)
    vol_ma20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else v

    turnover = float(c * v) if np.isfinite(c) and np.isfinite(v) else float("nan")
    adv20 = float((close * vol).rolling(20).mean().iloc[-1]) if len(close) >= 20 else turnover

    # SMA20 slope: 5営業日前との差（pct）
    if len(close) >= 25:
        sma20_series = close.rolling(20).mean()
        sma20_now = float(sma20_series.iloc[-1])
        sma20_prev = float(sma20_series.iloc[-6])
        sma20_slope_5d = float((sma20_now / sma20_prev - 1.0)) if sma20_prev > 0 else 0.0
    else:
        sma20_slope_5d = 0.0

    # RS 20d: (stock 20d return - index 20d return)
    rs_20d = 0.0
    try:
        if index_hist is not None and len(index_hist) >= 25 and len(close) >= 25:
            ic = index_hist["Close"].astype(float)
            stock_ret = float(close.iloc[-1] / close.iloc[-21] - 1.0)
            idx_ret = float(ic.iloc[-1] / ic.iloc[-21] - 1.0)
            rs_20d = float(stock_ret - idx_ret)
    except Exception:
        rs_20d = 0.0

    return FeaturePack(
        close=float(c),
        open=float(o),
        high=float(h),
        low=float(l),
        prev_close=float(pc),
        sma10=float(sma10),
        sma20=float(sma20),
        sma50=float(sma50),
        rsi14=float(rsi14),
        atr14=float(atr14),
        atr_pct=float(atr_pct),
        hh20=float(hh20),
        vol=float(v),
        vol_ma20=float(vol_ma20),
        turnover=float(turnover),
        adv20=float(adv20),
        sma20_slope_5d=float(sma20_slope_5d),
        rs_20d=float(rs_20d),
    )
