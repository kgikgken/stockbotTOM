　# ============================================
# utils/market.py
# 地合い判定・マーケットスコア算出
# ============================================

import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict

from utils.util import safe_div


# --------------------------------------------
# 設定
# --------------------------------------------
INDEX_TICKER = "^TOPX"   # TOPIX
LOOKBACK_DAYS = 60


# --------------------------------------------
# 移動平均
# --------------------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


# --------------------------------------------
# RSI
# --------------------------------------------
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --------------------------------------------
# マーケットデータ取得
# --------------------------------------------
def load_market_data() -> pd.DataFrame:
    df = yf.download(
        INDEX_TICKER,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("Market data is empty")

    return df.dropna()


# --------------------------------------------
# MarketScore 計算
# --------------------------------------------
def calc_market_score() -> Dict:
    """
    MarketScore:
        0〜100
    出力:
        {
            "score": float,
            "delta_3d": float,
            "trend": str,   # up / neutral / down
        }
    """

    df = load_market_data()
    close = df["Close"]

    # 移動平均
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)

    # RSI
    rsi14 = calc_rsi(close, 14)

    latest = df.iloc[-1]
    prev_3d = df.iloc[-4] if len(df) >= 4 else df.iloc[0]

    score = 50.0  # 基準点

    # -------------------------
    # トレンド構造
    # -------------------------
    if close.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]:
        score += 15
        trend = "up"
    elif close.iloc[-1] < sma50.iloc[-1]:
        score -= 15
        trend = "down"
    else:
        trend = "neutral"

    # -------------------------
    # RSI
    # -------------------------
    rsi = rsi14.iloc[-1]
    if rsi > 60:
        score += 5
    elif rsi < 40:
        score -= 5

    # -------------------------
    # 直近モメンタム
    # -------------------------
    ret_3d = safe_div(latest["Close"] - prev_3d["Close"], prev_3d["Close"])
    delta_3d = ret_3d * 100

    if delta_3d > 1.0:
        score += 5
    elif delta_3d < -1.0:
        score -= 5

    # -------------------------
    # 上限・下限
    # -------------------------
    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "delta_3d": round(delta_3d, 2),
        "trend": trend,
    }


# --------------------------------------------
# 新規可否判定（完全機械化）
# --------------------------------------------
def is_no_trade_day(market_score: float, delta_3d: float) -> bool:
    """
    NO-TRADE 条件
    """
    if market_score < 45:
        return True
    if delta_3d <= -5 and market_score < 55:
        return True
    return False