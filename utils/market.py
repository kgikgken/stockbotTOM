# ============================================
# utils/market.py
# 市場環境（地合い）判定
# ============================================

import pandas as pd
import numpy as np

# --------------------------------------------
# Market Regime 判定
# --------------------------------------------
def calc_market_regime(index_df: pd.DataFrame) -> dict:
    """
    市場の地合いを数値化して返す
    """
    if index_df is None or len(index_df) < 50:
        return {
            "score": 50,
            "delta_3d": 0,
            "trend": "neutral",
        }

    close = index_df["Close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    score = 50

    if close.iloc[-1] > ma50.iloc[-1]:
        score += 10
    if ma20.iloc[-1] > ma50.iloc[-1]:
        score += 10

    delta_3d = close.iloc[-1] - close.iloc[-4]

    if delta_3d > 0:
        score += 5
    else:
        score -= 5

    trend = "up" if score >= 60 else "down" if score <= 45 else "neutral"

    return {
        "score": int(score),
        "delta_3d": float(delta_3d),
        "trend": trend,
    }