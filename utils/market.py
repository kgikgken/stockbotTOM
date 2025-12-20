from __future__ import annotations

import numpy as np
import yfinance as yf


def _five_day_chg(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return 0.0
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return 0.0


def calc_market_score() -> dict:
    nk = _five_day_chg("^N225")
    tp = _five_day_chg("^TOPX")

    base = 50.0 + float(np.clip((nk + tp) / 2.0, -20, 20))
    score = int(np.clip(round(base), 0, 100))

    comment = (
        "強め" if score >= 70 else
        "やや強め" if score >= 60 else
        "中立" if score >= 50 else
        "弱め" if score >= 40 else
        "弱い"
    )

    return {"score": score, "comment": comment, "n225_5d": nk, "topix_5d": tp}


def enhance_market_score() -> dict:
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX
    try:
        sox = _five_day_chg("^SOX")
        score += float(np.clip(sox / 2.0, -5.0, 5.0))
        mkt["sox_5d"] = sox
    except Exception:
        pass

    # NVDA
    try:
        nvda = _five_day_chg("NVDA")
        score += float(np.clip(nvda / 3.0, -4.0, 4.0))
        mkt["nvda_5d"] = nvda
    except Exception:
        pass

    mkt["score"] = int(np.clip(round(score), 0, 100))
    return mkt
