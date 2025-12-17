from __future__ import annotations

import numpy as np
import yfinance as yf

def _five_day_chg(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return 0.0
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return 0.0

def calc_market_score() -> dict:
    """N225 + TOPIX 5d change -> 0..100 score."""
    nk = _five_day_chg("^N225")
    tp = _five_day_chg("^TOPX")

    base = 50.0 + float(np.clip((nk + tp) / 2.0, -20, 20))
    score = int(np.clip(round(base), 0, 100))

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

    return {"score": score, "comment": comment, "n225_5d": nk, "topix_5d": tp}

def enhance_market_score() -> dict:
    """calc_market_score + SOX/NVDA small overlay."""
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX overlay
    try:
        sox = yf.Ticker("^SOX").history(period="6d", auto_adjust=True)
        if sox is not None and not sox.empty and len(sox) >= 2:
            chg = float((sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0)
            score += float(np.clip(chg / 2.0, -5.0, 5.0))
            mkt["sox_5d"] = chg
    except Exception:
        pass

    # NVDA overlay
    try:
        nv = yf.Ticker("NVDA").history(period="6d", auto_adjust=True)
        if nv is not None and not nv.empty and len(nv) >= 2:
            chg = float((nv["Close"].iloc[-1] / nv["Close"].iloc[0] - 1.0) * 100.0)
            score += float(np.clip(chg / 3.0, -4.0, 4.0))
            mkt["nvda_5d"] = chg
    except Exception:
        pass

    score = int(np.clip(round(score), 0, 100))
    mkt["score"] = score
    return mkt
