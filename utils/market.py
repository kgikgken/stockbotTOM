# ============================================
# utils/market.py
# 地合い判定・マーケットスコア算出
# ============================================

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------
# MarketScore 計算
# --------------------------------------------
def calc_market_score(index_df: pd.DataFrame) -> dict:
    """
    index_df: 日付昇順の DataFrame（Close 必須）
    return:
        {
            "score": 0-100,
            "delta_3d": int,
            "trend": "up" | "flat" | "down"
        }
    """
    df = index_df.copy()

    # 必須チェック
    if "Close" not in df.columns or len(df) < 60:
        return {
            "score": 50,
            "delta_3d": 0,
            "trend": "flat",
        }

    close = df["Close"]

    # 移動平均
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    score = 50

    # トレンド構造
    if close.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-1] > sma50.iloc[-1]:
        score += 15
        trend = "up"
    elif close.iloc[-1] < sma50.iloc[-1]:
        score -= 15
        trend = "down"
    else:
        trend = "flat"

    # モメンタム（5日・20日）
    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100

    score += np.clip(ret_5d * 1.2, -10, 10)
    score += np.clip(ret_20d * 0.6, -10, 10)

    # ボラティリティ警戒（急落）
    daily_ret = close.pct_change()
    if daily_ret.iloc[-1] < -0.03:
        score -= 10

    score = int(np.clip(score, 0, 100))

    # Δ3日
    if len(df) >= 4:
        delta_3d = score - int(
            calc_market_score(df.iloc[:-3])["score"]
        )
    else:
        delta_3d = 0

    return {
        "score": score,
        "delta_3d": delta_3d,
        "trend": trend,
    }


# --------------------------------------------
# 新規可否判定
# --------------------------------------------
def is_no_trade_day(market_score: int, delta_3d: int) -> tuple[bool, str]:
    """
    NO-TRADE 条件
    """
    if market_score < 45:
        return True, "MarketScore<45"

    if delta_3d <= -5 and market_score < 55:
        return True, "Δ3d<=-5 & MarketScore<55"

    return False, ""