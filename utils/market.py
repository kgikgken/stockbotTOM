from __future__ import annotations

import numpy as np
import yfinance as yf


def _fetch(symbol: str, period: str = "10d"):
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _five_day_chg(symbol: str) -> float:
    df = _fetch(symbol, period="6d")
    if df is None or len(df) < 2:
        return 0.0
    c = df["Close"].astype(float)
    return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)


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
    score = float(mkt["score"])

    try:
        sox = _five_day_chg("^SOX")
        score += float(np.clip(sox / 2.0, -5, 5))
        mkt["sox_5d"] = sox
    except Exception:
        pass

    try:
        nvda = _five_day_chg("NVDA")
        score += float(np.clip(nvda / 3.0, -4, 4))
        mkt["nvda_5d"] = nvda
    except Exception:
        pass

    mkt["score"] = int(np.clip(round(score), 0, 100))
    return mkt


def market_score_delta_3d() -> int:
    today = calc_market_score().get("score", 50)

    def approx_chg_3d_ago(symbol: str) -> float:
        df = _fetch(symbol, period="10d")
        if df is None or len(df) < 6:
            return 0.0
        c = df["Close"].astype(float)
        i = -4
        j = max(i - 5, -len(c))
        base = float(c.iloc[j])
        now = float(c.iloc[i])
        if base <= 0:
            return 0.0
        return (now / base - 1.0) * 100.0

    nk = approx_chg_3d_ago("^N225")
    tp = approx_chg_3d_ago("^TOPX")

    base = 50.0 + float(np.clip((nk + tp) / 2.0, -20, 20))
    past = int(np.clip(round(base), 0, 100))

    return int(today) - int(past)
