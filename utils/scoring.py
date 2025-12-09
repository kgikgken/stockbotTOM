from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# 内部ヘルパー
# ============================================================
def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance history を受け取り、スコアに使う指標を付与
    """
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

    # 20日ボラ
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # 過去60日高値からの位置
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
    else:
        df["off_high_pct"] = np.nan

    # 出来高・売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


# ============================================================
# スコア構成
# ============================================================
def _trend_score(df: pd.DataFrame) -> float:
    close = df["Close"].astype(float)
    ma20 = df["ma20"]
    ma50 = df["ma50"]

    s_last = _last_val(ma20.pct_change(fill_method=None))
    c_last = _last_val(close)
    ma20_last = _last_val(ma20)
    ma50_last = _last_val(ma50)

    sc = 0.0

    # MA20の傾き
    if np.isfinite(s_last):
        if s_last >= 0.01:
            sc += 10
        elif s_last > 0:
            sc += 5 + (s_last / 0.01) * 5
        else:
            sc += max(0.0, 5 + s_last * 80)

    # MAの並び
    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last > ma50_last:
            sc += 10
        elif c_last > ma20_last:
            sc += 5
        elif ma20_last > ma50_last:
            sc += 3

    return float(np.clip(sc, 0, 20))


def _pullback_score(df: pd.DataFrame) -> float:
    rsi = _last_val(df["rsi14"])
    off = _last_val(df["off_high_pct"])

    sc = 0.0

    # RSI: 理想は30〜45
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 12
        elif 25 <= rsi < 30 or 45 < rsi <= 55:
            sc += 7
        elif 20 <= rsi < 25 or 55 < rsi <= 60:
            sc += 3

    # 高値からの下落率: -7〜-18% くらい
    if np.isfinite(off):
        if -18 <= off <= -7:
            sc += 12
        elif -25 <= off < -18 or -7 < off <= 0:
            sc += 6

    return float(np.clip(sc, 0, 24))


def _liquidity_score(df: pd.DataFrame) -> float:
    t = _last_val(df["turnover_avg20"])
    v = _last_val(df["vola20"])
    sc = 0.0

    # 流動性（売買代金）
    if np.isfinite(t):
        if t >= 3e8:
            sc += 12
        elif t >= 1e8:
            sc += 12 * (t - 1e8) / 2e8

    # ボラ（極端すぎない）
    if np.isfinite(v):
        if 0.015 <= v <= 0.06:
            sc += 6
        elif 0.01 <= v < 0.015 or 0.06 < v <= 0.08:
            sc += 3

    return float(np.clip(sc, 0, 18))


# ============================================================
# 外部公開API
# ============================================================
def score_stock(hist: pd.DataFrame) -> float | None:
    """
    銘柄のCoreスコア（0〜100）
    - トレンド
    - 押し目完成度
    - 流動性/ボラ
    """
    if hist is None or len(hist) < 60:
        return None

    df = _add_indicators(hist)

    ts = _trend_score(df)
    ps = _pullback_score(df)
    ls = _liquidity_score(df)

    raw = ts + ps + ls  # 最大 20 + 24 + 18 = 62
    if not np.isfinite(raw):
        return None

    score = float(raw / 62.0 * 100.0)
    return float(np.clip(score, 0, 100))