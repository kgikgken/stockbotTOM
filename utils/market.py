from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import sma, rsi14, atr

def _fetch_index(symbol: str, period: str = "180d") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty or len(df) < 80:
            return None
        return df
    except Exception:
        return None

def _score_from_df(df: pd.DataFrame, idx: int) -> int:
    close = df["Close"].astype(float)
    c = float(close.iloc[idx])
    sub = df.iloc[: idx + 1].copy()
    csub = sub["Close"].astype(float)

    ma20 = sma(csub, 20)
    ma50 = sma(csub, 50)
    rsi = rsi14(csub)
    a = atr(sub, 14)

    ma20v = float(ma20.iloc[-1]) if len(ma20) else c
    ma50v = float(ma50.iloc[-1]) if len(ma50) else c
    rsiv = float(rsi.iloc[-1]) if len(rsi) else 50.0
    atrv = float(a.iloc[-1]) if len(a) else (c * 0.015)

    trend = 0.2
    if np.isfinite(ma20v) and np.isfinite(ma50v):
        if c > ma20v > ma50v:
            trend = 1.0
        elif c > ma20v:
            trend = 0.6
        elif ma20v > ma50v:
            trend = 0.4

    mom = 0.5
    if np.isfinite(rsiv):
        if rsiv >= 60:
            mom = 0.8
        elif rsiv >= 50:
            mom = 0.6
        elif rsiv >= 40:
            mom = 0.45
        else:
            mom = 0.3

    atr_pct = atrv / c if (np.isfinite(atrv) and np.isfinite(c) and c > 0) else 0.02
    risk = 0.6
    if atr_pct >= 0.03:
        risk = 0.4
    if atr_pct >= 0.05:
        risk = 0.25

    s = 50.0
    s += (trend - 0.5) * 30.0
    s += (mom - 0.5) * 25.0
    s += (risk - 0.5) * 20.0

    if len(csub) >= 6:
        r5 = float(csub.iloc[-1] / csub.iloc[-6] - 1.0)
        s += float(np.clip(r5 * 100.0, -6.0, 6.0))

    return int(np.clip(round(s), 0, 100))

def enhance_market_score() -> Dict:
    topix = _fetch_index("^TOPX")
    nikkei = _fetch_index("^N225")

    if topix is None and nikkei is None:
        return {"score": 50, "comment": "中立", "delta3d": 0}

    def _calc(df: pd.DataFrame) -> Tuple[int, int]:
        now = _score_from_df(df, -1)
        prev = _score_from_df(df, -4) if len(df) >= 10 else now
        return now, now - prev

    scores = []
    deltas = []
    for df in (topix, nikkei):
        if df is None:
            continue
        s, d = _calc(df)
        scores.append(s)
        deltas.append(d)

    score = int(np.clip(round(float(np.mean(scores))), 0, 100))
    delta3d = int(np.clip(round(float(np.mean(deltas))), -20, 20))

    if score >= 70:
        comment = "強め"
    elif score >= 60:
        comment = "やや強め"
    elif score >= 50:
        comment = "中立"
    elif score >= 40:
        comment = "弱め"
    else:
        comment = "弱い"

    return {"score": score, "comment": comment, "delta3d": delta3d}
