from __future__ import annotations

import numpy as np
import pandas as pd


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    日足ATR（シンプル）
    hist_d は yfinance auto_adjust=True の日足を想定（Open/High/Low/Closeがある）
    """
    if df is None or len(df) < period + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else np.nan


def score_daytrade_candidate(hist_d: pd.DataFrame, mkt_score: int = 50) -> float:
    """
    v11.1+ 寄り後Day用（ログ無し最終形）
    「押し戻し → 再上昇」だけを高評価し、初動完璧（=危険）を除外する。

    出力: 0〜100
    """
    if hist_d is None or len(hist_d) < 80:
        return 0.0

    df = hist_d.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    atr = _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        return 0.0

    c = _last(close)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    m20 = _last(ma20)
    m60 = _last(ma60)

    # 1) トレンド前提
    trend_ok = bool(np.isfinite(c) and np.isfinite(m20) and np.isfinite(m60) and c > m20 > m60)

    # 2) 初動強すぎ除外（GU一気上げ/天井圏に乗るのを防ぐ）
    if len(close) >= 3:
        surge = float(close.iloc[-1] - close.iloc[-3])
        if surge >= 1.8 * atr:
            return 0.0

    # 3) 押し戻し（直近5日高値→0.4〜1.2ATR押し）
    recent_high = float(high.iloc[-6:-1].max())
    pullback = float(recent_high - c)
    pullback_atr = pullback / atr if atr > 0 else 999.0
    pullback_ok = bool(0.4 <= pullback_atr <= 1.2)

    # 4) 再上昇の兆し（安値切り上げ）
    reup = bool(len(low) >= 3 and float(low.iloc[-1]) > float(low.iloc[-2]))

    # 5) 出来高（売買代金）上昇
    value_now = float(vol.iloc[-1] * close.iloc[-1]) if np.isfinite(vol.iloc[-1]) and np.isfinite(close.iloc[-1]) else np.nan
    value_ma20 = float((vol * close).rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
    vol_ok = bool(np.isfinite(value_now) and np.isfinite(value_ma20) and value_ma20 > 0 and value_now >= value_ma20)

    sc = 0.0
    if trend_ok:
        sc += 25.0
    if pullback_ok:
        sc += 30.0
    if reup:
        sc += 20.0
    if vol_ok:
        sc += 15.0

    # 地合い補正（Dayは影響大）
    sc += float((mkt_score - 50) * 0.30)

    return float(np.clip(sc, 0, 100))
