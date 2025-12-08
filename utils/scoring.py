from __future__ import annotations
import numpy as np
import pandas as pd

# ============================================================
# 内部ヘルパー：インジケーター計算（ACDE専用）
# ============================================================
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    hist（yfinanceのhistory）を受け取り、
    ACDEで使う指標を載せる。
    """
    df = df.copy()

    close = df["Close"].astype(float)
    high = df.get("High", close).astype(float)
    low = df.get("Low", close).astype(float)
    vol = df.get("Volume", pd.Series([0]*len(df))).astype(float)

    # MA
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ボラ20
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # 60日高値距離
    if len(close) >= 60:
        h60 = close.rolling(60).max()
        df["off_high_pct"] = (close - h60) / h60 * 100
    else:
        df["off_high_pct"] = np.nan

    # MA20傾き
    df["trend_slope20"] = df["ma20"].pct_change(fill_method=None)

    # 流動性（売買代金）
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


def _last(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


# ============================================================
# Trend（0〜20）
# ============================================================
def _score_trend(df: pd.DataFrame) -> float:
    close = df["Close"].astype(float)
    ma20 = df["ma20"]
    ma50 = df["ma50"]
    slope = df["trend_slope20"]

    c = _last(close)
    m20 = _last(ma20)
    m50 = _last(ma50)
    s = _last(slope)

    sc = 0.0

    # MA並び
    if np.isfinite(c) and np.isfinite(m20) and np.isfinite(m50):
        if c > m20 > m50:
            sc += 10
        elif c > m20:
            sc += 6
        elif m20 > m50:
            sc += 3

    # 傾き（トレンド速度）
    if np.isfinite(s):
        if s >= 0.01:
            sc += 10
        elif s > 0:
            # 線形：0〜0.01 → 0〜10
            sc += (s / 0.01) * 10
        else:
            sc += max(0.0, 10 + s * 120)  # -0.08でほぼ0

    return float(np.clip(sc, 0, 20))


# ============================================================
# Pullback（0〜20）
# ============================================================
def _score_pullback(df: pd.DataFrame) -> float:
    rsi = _last(df["rsi14"])
    off = _last(df["off_high_pct"])

    sc = 0.0

    # RSI
    if np.isfinite(rsi):
        if 38 <= rsi <= 52:
            sc += 10
        elif 32 <= rsi < 38 or 52 < rsi <= 60:
            sc += 6
        else:
            sc += 2

    # 高値からの押し幅
    if np.isfinite(off):
        # 理想押し目：-12〜-5%
        if -12 <= off <= -5:
            sc += 10
        # 浅め or 深め
        elif -4 <= off <= 0 or -20 <= off < -12:
            sc += 5
        else:
            sc += 1

    return float(np.clip(sc, 0, 20))


# ============================================================
# Liquidity（0〜20）
# ============================================================
def _score_liquidity(df: pd.DataFrame) -> float:
    t = _last(df["turnover_avg20"])
    v = _last(df["vola20"])

    sc = 0.0

    # 流動性（売買代金）
    if np.isfinite(t):
        if t >= 10e8:
            sc += 16
        elif t >= 1e8:
            sc += 16 * (t - 1e8) / 9e8

    # ボラ（極端でなければ加点）
    if np.isfinite(v):
        if v < 0.02:
            sc += 4
        elif v < 0.06:
            sc += 4 * (0.06 - v) / 0.04

    return float(np.clip(sc, 0, 20))


# ============================================================
# Public API
# ============================================================
def score_stock(hist: pd.DataFrame) -> float | None:
    """
    Coreスコア（0〜100）
    Aランク: >=80
    Bランク: >=70
    """
    if hist is None or len(hist) < 60:
        return None

    df = _add_indicators(hist)

    trend = _score_trend(df)        # 0〜20
    pullback = _score_pullback(df)  # 0〜20
    liquidity = _score_liquidity(df)# 0〜20

    raw = trend + pullback + liquidity  # 最大60
    if not np.isfinite(raw):
        return None

    score = float(raw / 60 * 100)  # 0〜100
    return float(np.clip(score, 0, 100))