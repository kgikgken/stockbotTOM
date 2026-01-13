from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import sma, returns, safe_float, clamp

def _fetch_index(symbol: str, period: str = "220d") -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _ma_structure_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 60:
        return 0.0
    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    c_last = safe_float(c.iloc[-1], np.nan)
    m20 = safe_float(ma20.iloc[-1], np.nan)
    m50 = safe_float(ma50.iloc[-1], np.nan)
    if not (np.isfinite(c_last) and np.isfinite(m20) and np.isfinite(m50)):
        return 0.0

    sc = 0.0
    if c_last > m20 > m50:
        sc += 12
    elif c_last > m20:
        sc += 6
    elif m20 > m50:
        sc += 3

    slope20 = safe_float(ma20.pct_change(fill_method=None).iloc[-1], 0.0)
    if slope20 >= 0.004:
        sc += 8
    elif slope20 > 0:
        sc += 4 + slope20 / 0.004 * 4
    else:
        sc += max(0.0, 4 + slope20 * 200)

    return float(clamp(sc, 0, 20))

def _momentum_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 21:
        return 0.0
    c = df["Close"].astype(float)
    r5 = safe_float(c.iloc[-1] / c.iloc[-6] - 1.0, 0.0) * 100.0
    r20 = safe_float(c.iloc[-1] / c.iloc[-21] - 1.0, 0.0) * 100.0
    sc = 0.0
    sc += clamp(r5, -6, 6) * 2.0
    sc += clamp(r20, -12, 12) * 1.0
    return float(clamp(sc, -25, 25))

def _vol_gap_penalty(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 30:
        return 0.0
    c = df["Close"].astype(float)
    o = df["Open"].astype(float)
    prev = c.shift(1)
    gap = ((o - prev).abs() / (prev + 1e-9))
    gap_freq = float((gap.tail(20) > 0.012).mean())
    vola = safe_float(returns(df).tail(20).std(), 0.0)
    pen = 0.0
    pen += gap_freq * 10.0
    pen += clamp((vola - 0.012) * 500.0, 0.0, 10.0)
    return float(-pen)

def market_score() -> Dict[str, float]:
    n225 = _fetch_index("^N225")
    topx = _fetch_index("^TOPX")

    base = 50.0
    base += _ma_structure_score(n225)
    base += _ma_structure_score(topx)
    base += _momentum_score(n225) * 0.6
    base += _momentum_score(topx) * 0.6
    base += (_vol_gap_penalty(n225) + _vol_gap_penalty(topx)) * 0.7

    score = int(clamp(round(base), 0, 100))

    if score >= 70:
        comment = "強い"
    elif score >= 60:
        comment = "やや強い"
    elif score >= 50:
        comment = "中立"
    elif score >= 40:
        comment = "弱い"
    else:
        comment = "かなり弱い"

    return {"score": float(score), "comment": comment}

def futures_risk_on() -> Tuple[bool, float]:
    try:
        df = yf.Ticker("NKD=F").history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return False, 0.0
        chg = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0
        return bool(chg >= 1.0), float(chg)
    except Exception:
        return False, 0.0
