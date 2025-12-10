from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ======================================
# インジケータ作成
# ======================================
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
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

    # 20日ボラ（標準偏差）
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # 60日高値からの距離
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
    else:
        df["off_high_pct"] = np.nan

    # 出来高トレンド
    df["vol_ma20"] = vol.rolling(20).mean()

    return df


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


# ======================================
# スコアリング本体
# ======================================
def score_stock(ticker: str, hist: pd.DataFrame, uni_row: Any | None = None) -> float:
    """
    テクニカル条件だけで 0-100 点のスコアを返す。
    """
    if hist is None or len(hist) < 60:
        return 0.0

    df = _add_indicators(hist)
    score = 0.0

    # 1. トレンド（上昇 & 初押し寄りを評価）
    ma20 = df["ma20"]
    ma50 = df["ma50"]
    close = df["Close"].astype(float)

    slope20 = ma20.pct_change(fill_method=None).iloc[-5:].mean()
    above_ma = close.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]

    if np.isfinite(slope20) and slope20 > 0:
        score += 20.0
    if above_ma:
        score += 10.0

    # 2. RSIゾーン評価
    rsi = _last(df["rsi14"])
    if 30 <= rsi <= 55:
        score += 20.0
    elif 25 <= rsi < 30 or 55 < rsi <= 60:
        score += 10.0

    # 3. 高値からの押し具合
    off_high = _last(df["off_high_pct"])
    if np.isfinite(off_high):
        if -20 <= off_high <= -5:
            score += 15.0
        elif -30 <= off_high < -20:
            score += 8.0

    # 4. ボラ（極端に低すぎ/高すぎは減点）
    vola = _last(df["vola20"])
    if np.isfinite(vola):
        if 0.015 <= vola <= 0.06:
            score += 15.0
        elif 0.01 <= vola < 0.015 or 0.06 < vola <= 0.09:
            score += 7.0

    # 5. 出来高
    vol = hist["Volume"].astype(float)
    vol_ma20 = df["vol_ma20"]
    vol_ratio = vol.iloc[-1] / (vol_ma20.iloc[-1] + 1e-9)
    if vol_ratio >= 2.0:
        score += 10.0
    elif vol_ratio >= 1.2:
        score += 5.0

    # 0-100 にクリップ
    return float(np.clip(score, 0.0, 100.0))