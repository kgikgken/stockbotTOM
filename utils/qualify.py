from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import _last_val

def _pct(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return 0.0
    return float((a / b - 1.0) * 100.0)

def runner_strength(hist: pd.DataFrame) -> float:
    """0..100: '走行能力' proxy (trend + momentum + liquidity)."""
    if hist is None or len(hist) < 220:
        return 0.0
    close = hist["Close"].astype(float)
    vol = hist["Volume"].astype(float) if "Volume" in hist.columns else pd.Series(np.nan, index=hist.index)

    c = _last_val(close)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])

    # 20d momentum
    if len(close) >= 21:
        mom20 = _pct(close.iloc[-1], close.iloc[-21])
    else:
        mom20 = 0.0

    # 60d momentum
    if len(close) >= 61:
        mom60 = _pct(close.iloc[-1], close.iloc[-61])
    else:
        mom60 = 0.0

    # Liquidity
    turnover20 = float((close * vol).rolling(20).mean().iloc[-1]) if len(close) >= 20 else 0.0

    sc = 0.0
    # Structure
    if c > ma20 > ma60 > ma200:
        sc += 45.0
    elif c > ma20 > ma60:
        sc += 30.0
    elif c > ma20:
        sc += 18.0

    sc += float(np.clip(mom20 * 2.0, -10, 20))
    sc += float(np.clip(mom60 * 1.2, -10, 20))

    if np.isfinite(turnover20):
        if turnover20 >= 1e9:
            sc += 20
        elif turnover20 >= 1e8:
            sc += 20 * (turnover20 - 1e8) / 9e8

    return float(np.clip(sc, 0, 100))

def runner_class(strength: float) -> str:
    if strength >= 85:
        return "A2_prebreak"
    if strength >= 75:
        return "A3_trend"
    if strength >= 65:
        return "B_trend"
    return "C"

def al3_score(base_score: float, rr: float, ev_r: float, runner_strength_v: float) -> float:
    """AL3 numeric score (higher = better) used for ranking top5."""
    if not all(np.isfinite(x) for x in (base_score, rr, ev_r, runner_strength_v)):
        return 0.0
    # Weights tuned for '勝ちたい' = keep edge + avoid junk
    return float(
        0.40 * (base_score / 100.0) +
        0.25 * np.clip(rr / 6.0, 0, 2) +
        0.20 * np.clip(ev_r / 1.2, 0, 2) +
        0.15 * (runner_strength_v / 100.0)
    ) * 10.0  # make it readable (~0..20)
