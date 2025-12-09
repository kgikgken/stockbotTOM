from __future__ import annotations

import numpy as np
import yfinance as yf


def _five_day_chg(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="5d")
        if df is None or df.empty:
            return 0.0
        return float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0
    except Exception:
        return 0.0


def calc_market_score() -> dict:
    """
    日経平均・TOPIXの5日リターンからベース市場スコアを計算。
    戻り値: {"score": int, "comment": str}
    """
    nk = _five_day_chg("^N225")
    tp = _five_day_chg("^TOPX")

    base = 50.0
    base += np.clip((nk + tp) / 2.0, -20.0, 20.0) * 1.0

    score = int(np.clip(round(base), 0, 100))

    if score >= 70:
        comment = "地合い強め"
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
    """
    calc_market_scoreに
    ・SOX指数
    ・NVDA
    の5日リターンをブーストとして追加。
    """
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX
    try:
        df_sox = yf.Ticker("^SOX").history(period="5d")
        if df_sox is not None and not df_sox.empty:
            sox_chg = float(df_sox["Close"].iloc[-1] / df_sox["Close"].iloc[0] - 1.0) * 100.0
            score += np.clip(sox_chg / 2.0, -5.0, 5.0)
    except Exception:
        pass

    # NVDA
    try:
        df_nvda = yf.Ticker("NVDA").history(period="5d")
        if df_nvda is not None and not df_nvda.empty:
            nvda_chg = float(df_nvda["Close"].iloc[-1] / df_nvda["Close"].iloc[0] - 1.0) * 100.0
            score += np.clip(nvda_chg / 3.0, -4.0, 4.0)
    except Exception:
        pass

    score = int(np.clip(round(score), 0, 100))
    mkt["score"] = score
    return mkt