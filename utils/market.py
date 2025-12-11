from __future__ import annotations

import numpy as np
import yfinance as yf


def _five_day_chg(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="6d")
        if df is None or df.empty or len(df) < 2:
            return 0.0
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return 0.0


def calc_market_score() -> dict:
    nk = _five_day_chg("^N225")
    tp = _five_day_chg("^TOPX")
    base = 50.0 + np.clip((nk + tp) / 2.0, -20.0, 20.0)

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

    return {"score": score, "comment": comment}


def enhance_market_score() -> dict:
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # 半導体指数 SOX
    try:
        sox_chg = _five_day_chg("^SOX")
        score += np.clip(sox_chg / 2.0, -5.0, 5.0)
    except Exception:
        pass

    # NVDA
    try:
        nvda_chg = _five_day_chg("NVDA")
        score += np.clip(nvda_chg / 3.0, -4.0, 4.0)
    except Exception:
        pass

    score = int(np.clip(round(score), 0, 100))
    mkt["score"] = score
    return mkt