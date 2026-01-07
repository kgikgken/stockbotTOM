# ============================================
# utils/features.py
# テクニカル特徴量の算出
# ============================================

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------
# 移動平均
# --------------------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


# --------------------------------------------
# ATR（14）
# --------------------------------------------
def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - close).abs(),
            (low - close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window).mean()


# --------------------------------------------
# RSI（14）
# --------------------------------------------
def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


# --------------------------------------------
# 高値ブレイク判定
# --------------------------------------------
def highest(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()


# --------------------------------------------
# トレンド構造判定
# --------------------------------------------
def trend_structure(df: pd.DataFrame) -> dict:
    """
    MA構造と傾きをまとめて返す
    """
    close = df["Close"]

    ma20 = sma(close, 20)
    ma50 = sma(close, 50)

    slope20 = ma20.diff(5)

    return {
        "ma20": ma20.iloc[-1],
        "ma50": ma50.iloc[-1],
        "slope20": slope20.iloc[-1],
        "trend_up": ma20.iloc[-1] > ma50.iloc[-1] and slope20.iloc[-1] > 0,
    }


# --------------------------------------------
# 押し目距離（ATR基準）
# --------------------------------------------
def pullback_atr(close: float, ma20: float, atr: float) -> float:
    if atr <= 0:
        return 999.0
    return abs(close - ma20) / atr