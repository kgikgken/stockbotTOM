from __future__ import annotations

from typing import Dict

import numpy as np
import yfinance as yf


def _five_day_chg(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="5d")
        if df is None or df.empty or len(df) < 2:
            return 0.0
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return 0.0


def calc_market_score() -> Dict[str, float]:
    """
    日経平均・TOPIX ベースの地合いスコア
    """
    nk = _five_day_chg("^N225")
    tp = _five_day_chg("^TOPX")

    base = 50.0
    avg = (nk + tp) / 2.0
    base += np.clip(avg, -20.0, 20.0) * 1.0

    score = int(np.clip(round(base), 0, 100))

    if score >= 70:
        comment = "強い"
    elif score >= 60:
        comment = "やや強め"
    elif score >= 50:
        comment = "中立"
    elif score >= 40:
        comment = "弱め"
    else:
        comment = "弱い"

    return {"score": score, "comment": comment}


def enhance_market_score() -> Dict[str, float]:
    """
    calc_market_score に SOX と NVDA の情報を足して強化したスコア。
    """
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX
    try:
        sox = yf.Ticker("^SOX").history(period="5d")
        if sox is not None and not sox.empty and len(sox) >= 2:
            close = sox["Close"].astype(float)
            chg = float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
            score += np.clip(chg / 2.0, -5.0, 5.0)
    except Exception:
        pass

    # NVDA
    try:
        nvda = yf.Ticker("NVDA").history(period="5d")
        if nvda is not None and not nvda.empty and len(nvda) >= 2:
            close = nvda["Close"].astype(float)
            chg = float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
            score += np.clip(chg / 3.0, -4.0, 4.0)
    except Exception:
        pass

    score = int(np.clip(round(score), 0, 100))
    mkt["score"] = score

    # コメントは元のロジックのまま
    return mkt