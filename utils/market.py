from __future__ import annotations
import numpy as np
import yfinance as yf

def _five_day(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="6d", auto_adjust=True)
        if len(df) < 2:
            return 0.0
        return (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    except Exception:
        return 0.0

def calc_market_score() -> dict:
    nk = _five_day("^N225")
    tp = _five_day("^TOPX")

    score = 50 + (nk + tp) / 2
    score = int(np.clip(score, 0, 100))

    if score >= 70:
        c = "強め"
    elif score >= 60:
        c = "やや強め"
    elif score >= 50:
        c = "中立"
    elif score >= 40:
        c = "弱め"
    else:
        c = "弱い"

    return {"score": score, "comment": c}

def enhance_market_score() -> dict:
    m = calc_market_score()
    score = float(m["score"])

    try:
        sox = _five_day("^SOX")
        score += np.clip(sox / 2, -5, 5)
    except Exception:
        pass

    try:
        nv = _five_day("NVDA")
        score += np.clip(nv / 3, -4, 4)
    except Exception:
        pass

    m["score"] = int(np.clip(score, 0, 100))
    return m