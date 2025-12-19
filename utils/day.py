from __future__ import annotations

import numpy as np
import pandas as pd


def score_daytrade_candidate(hist_d: pd.DataFrame, mkt_score: int = 50) -> float:
    if hist_d is None or len(hist_d) < 80:
        return 0.0

    close = hist_d["Close"].astype(float)
    high = hist_d["High"].astype(float)
    low = hist_d["Low"].astype(float)
    vol = hist_d["Volume"].astype(float)

    atr = (high - low).rolling(14).mean().iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        return 0.0

    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    c = close.iloc[-1]

    trend_ok = c > ma20 > ma60
    recent_high = high.iloc[-6:-1].max()
    pullback = (recent_high - c) / atr
    pullback_ok = 0.4 <= pullback <= 1.2

    reup = low.iloc[-1] > low.iloc[-2]
    value_now = vol.iloc[-1] * close.iloc[-1]
    value_ma20 = (vol * close).rolling(20).mean().iloc[-1]
    vol_ok = value_now >= value_ma20

    sc = 0.0
    if trend_ok:
        sc += 25
    if pullback_ok:
        sc += 30
    if reup:
        sc += 20
    if vol_ok:
        sc += 15

    sc += (mkt_score - 50) * 0.3
    return float(np.clip(sc, 0, 100))