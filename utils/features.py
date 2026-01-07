# ============================================
# utils/features.py
# 特徴量計算（テクニカル / 環境）
# - MA / RSI / ATR / 出来高
# - Setup A1 / A2 判定用の基礎特徴
# ============================================

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------
# 基本テクニカル
# --------------------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# --------------------------------------------
# 出来高指標
# --------------------------------------------
def volume_ma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).mean()


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    vol_ma = volume_ma(volume, window)
    return volume / vol_ma.replace(0, np.nan)


# --------------------------------------------
# MA 構造判定
# --------------------------------------------
def ma_structure(close: pd.Series) -> pd.DataFrame:
    """
    Close, SMA20, SMA50, SMA200 を返す
    """
    df = pd.DataFrame(index=close.index)
    df["close"] = close
    df["sma20"] = sma(close, 20)
    df["sma50"] = sma(close, 50)
    df["sma200"] = sma(close, 200)
    return df


def ma_slope(series: pd.Series, lookback: int = 5) -> float:
    """
    MA の傾き（直近差分）
    """
    if len(series) < lookback + 1:
        return 0.0
    return series.iloc[-1] - series.iloc[-1 - lookback]


# --------------------------------------------
# ボラ・流動性
# --------------------------------------------
def atr_percent(atr_val: float, price: float) -> float:
    if price <= 0:
        return 0.0
    return atr_val / price * 100.0


def turnover(volume: float, price: float) -> float:
    """
    売買代金（概算）
    """
    return volume * price


# --------------------------------------------
# Setup A1 / A2 判定用特徴
# --------------------------------------------
def setup_a_features(df: pd.DataFrame) -> dict:
    """
    Setup A 系で使う特徴量をまとめて返す
    """
    close = df["Close"]
    volume = df["Volume"]

    ma = ma_structure(close)
    atr14 = atr(df, 14)
    rsi14 = rsi(close, 14)
    vol_ratio = volume_ratio(volume, 20)

    latest = {
        "close": close.iloc[-1],
        "sma20": ma["sma20"].iloc[-1],
        "sma50": ma["sma50"].iloc[-1],
        "sma200": ma["sma200"].iloc[-1],
        "sma20_slope": ma_slope(ma["sma20"]),
        "atr": atr14.iloc[-1],
        "atr_pct": atr_percent(atr14.iloc[-1], close.iloc[-1]),
        "rsi": rsi14.iloc[-1],
        "vol_ratio": vol_ratio.iloc[-1],
    }

    return latest


# --------------------------------------------
# ブレイク判定用
# --------------------------------------------
def highest_high(series: pd.Series, window: int = 20) -> float:
    if len(series) < window:
        return series.max()
    return series.iloc[-window:].max()


def is_breakout(close: pd.Series, window: int = 20) -> bool:
    hh = highest_high(close, window)
    return close.iloc[-1] > hh