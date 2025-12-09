from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
    else:
        df["off_high_pct"] = np.nan

    return df


def score_stock(ticker: str, hist: pd.DataFrame, uni_row) -> float:
    """
    Coreスコア（0〜100）
    """
    if hist is None or len(hist) < 60:
        return 0.0

    df = _add_indicators(hist)

    score = 0.0

    # Trend（20MAの傾き）
    slope = _last_val(df["ma20"].pct_change(fill_method=None))
    if np.isfinite(slope) and slope > 0:
        score += 10
    elif np.isfinite(slope):
        score += max(0.0, 10 + slope * 100)

    # RSI
    rsi = _last_val(df["rsi14"])
    if 30 <= rsi <= 50:
        score += 6
    elif 20 <= rsi < 30 or 50 < rsi <= 60:
        score += 3

    # 高値からの押し
    off = _last_val(df["off_high_pct"])
    if np.isfinite(off):
        if -15 <= off <= 5:
            score += 4
        elif -25 <= off < -15:
            score += 2

    # 流動性（単純売買代金）
    price = _last_val(df["Close"])
    vol = _last_val(df["Volume"])
    t = price * vol
    if np.isfinite(t):
        if t >= 1e8:
            score += 10
        elif t >= 1e7:
            score += 10 * (t - 1e7) / 9e7

    return float(np.clip(score, 0, 100))