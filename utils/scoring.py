from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return float("nan")


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    # 移動平均
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 60日高値からの位置
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0
    else:
        df["off_high_pct"] = np.nan

    # 出来高・売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラ（20日）
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    return df


def score_stock(ticker: str, hist: pd.DataFrame, uni_row: Optional[pd.Series] = None) -> float:
    """
    銘柄のCoreスコア（0〜100）。
    trend / pullback / liquidity をミックス。
    """
    if hist is None or len(hist) < 60:
        return 0.0

    df = _add_indicators(hist)

    score = 0.0

    # Trend: 20MAの傾き
    slope = _last_val(df["ma20"].pct_change(fill_method=None))
    if np.isfinite(slope):
        if slope > 0:
            score += 10.0
        else:
            score += max(0.0, 10.0 + slope * 80.0)  # 緩やかな下げは少しだけ減点

    # Pullback: RSI + 高値からの押し
    rsi = _last_val(df["rsi14"])
    off = _last_val(df["off_high_pct"])
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            score += 8.0
        elif 20 <= rsi < 30 or 45 < rsi <= 60:
            score += 4.0
    if np.isfinite(off):
        if -18 <= off <= -5:
            score += 8.0
        elif -25 <= off < -18:
            score += 4.0

    # Liquidity: 売買代金
    t = _last_val(df["turnover_avg20"])
    if np.isfinite(t):
        if t >= 1e8:
            score += 12.0
        elif t >= 2e7:
            score += 12.0 * (t - 2e7) / (1e8 - 2e7)

    # Volatility penalty: 極端な低ボラ/高ボラは少し減点
    vola = _last_val(df["vola20"])
    if np.isfinite(vola):
        if vola < 0.01:
            score -= 2.0
        elif vola > 0.06:
            score -= 3.0

    score = float(np.clip(score, 0.0, 100.0))
    return score