from __future__ import annotations

from typing import Dict

import numpy as np
import yfinance as yf


def _five_day_change(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="5d")
        if df is None or df.empty:
            return 0.0
        close = df["Close"].astype(float)
        return float(close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
    except Exception:
        return 0.0


def enhance_market_score() -> Dict:
    """
    日経225 / TOPIX + SOX + NVDA から地合いスコアを作成
    0〜100 にクリップして返す
    """
    nk = _five_day_change("^N225")
    tp = _five_day_change("^TOPX")
    sox = _five_day_change("^SOX")
    nvda = _five_day_change("NVDA")

    base = 50.0

    jp_avg = (nk + tp) / 2.0
    base += np.clip(jp_avg, -15.0, 15.0) * 0.8

    semi_boost = np.clip(sox / 4.0, -5.0, 5.0)
    nvda_boost = np.clip(nvda / 5.0, -4.0, 4.0)

    score = base + semi_boost + nvda_boost
    score = int(float(np.clip(round(score), 0, 100)))

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

    return {
        "score": score,
        "comment": comment,
        "detail": {
            "nk225_5d": nk,
            "topix_5d": tp,
            "sox_5d": sox,
            "nvda_5d": nvda,
        },
    }