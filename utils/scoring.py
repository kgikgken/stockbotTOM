from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# 内部：指標を付与
# ============================================================
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance: history を受け取り、
    スコア計算に使う指標をまとめて付与する。
    """
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    # --- MA ---
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # --- RSI14 ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # --- vola20 ---
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # --- 60日高値からの位置 ---
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
        tail = close.tail(60)
        idx = int(np.argmax(tail.values))
        days_since_high60 = (len(tail) - 1) - idx
        df["days_since_high60"] = np.nan
        df.loc[df.index[-1], "days_since_high60"] = float(days_since_high60)
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    # --- 20MAの傾き ---
    df["trend_slope20"] = df["ma20"].pct_change(fill_method=None)

    # --- 下ヒゲ比率 ---
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    # --- 売買代金（流動性） ---
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
# Trendスコア（20点）
# ============================================================
def _trend_score(df: pd.DataFrame) -> float:
    close = df["Close"].astype(float)
    ma20 = df["ma20"]
    ma50 = df["ma50"]
    slope = df["trend_slope20"]

    sc = 0.0
    s = _last(slope)
    c = _last(close)
    m20 = _last(ma20)
    m50 = _last(ma50)

    # 傾き
    if np.isfinite(s):
        if s >= 0.01:
            sc += 8
        elif s > 0:
            sc += 4 + (s / 0.01) * 4
        else:
            sc += max(0.0, 4 + s * 50)

    # MA構造
    if np.isfinite(c) and np.isfinite(m20) and np.isfinite(m50):
        if c > m20 > m50:
            sc += 8
        elif c > m20:
            sc += 4
        elif m20 > m50:
            sc += 2

    # 高値からの距離（戻りの浅さ/深さ）
    off = _last(df["off_high_pct"])
    if np.isfinite(off):
        if off >= -5:
            sc += 4
        elif off >= -15:
            sc += 4 - abs(off + 5) * 0.2

    return float(np.clip(sc, 0, 20))


# ============================================================
# Pullbackスコア（20点）
# ============================================================
def _pullback_score(df: pd.DataFrame) -> float:
    rsi = _last(df["rsi14"])
    off = _last(df["off_high_pct"])
    days = _last(df["days_since_high60"])
    shadow = _last(df["lower_shadow_ratio"])

    sc = 0.0

    # RSI
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 7
        elif 20 <= rsi < 30 or 45 < rsi <= 55:
            sc += 4
        else:
            sc += 1

    # 高値からの下落率
    if np.isfinite(off):
        if -12 <= off <= -5:
            sc += 6
        elif -20 <= off < -12:
            sc += 3
        else:
            sc += 1

    # 日柄
    if np.isfinite(days):
        if 2 <= days <= 10:
            sc += 4
        elif 1 <= days < 2 or 10 < days <= 20:
            sc += 2

    # 下ヒゲ
    if np.isfinite(shadow):
        if shadow >= 0.5:
            sc += 3
        elif shadow >= 0.3:
            sc += 1

    return float(np.clip(sc, 0, 20))


# ============================================================
# Liquidityスコア（20点）
# ============================================================
def _liquidity_score(df: pd.DataFrame) -> float:
    t = _last(df["turnover_avg20"])
    v = _last(df["vola20"])
    sc = 0.0

    # 流動性
    if np.isfinite(t):
        if t >= 1e9:            # >=10億/日
            sc += 16
        elif t >= 1e8:
            sc += 16 * (t - 1e8) / 9e8

    # ボラ（安定性）
    if np.isfinite(v):
        if v < 0.02:
            sc += 4
        elif v < 0.06:
            sc += 4 * (0.06 - v) / 0.04

    return float(np.clip(sc, 0, 20))


# ============================================================
# 公開：score_stock（0〜100）
# ============================================================
def score_stock(hist: pd.DataFrame) -> float | None:
    """
    Coreスコア：0〜100

    Aランク: 80以上
    Bランク: 70以上80未満
    """
    if hist is None or len(hist) < 60:
        return None

    df = _add_indicators(hist)

    ts = _trend_score(df)
    ps = _pullback_score(df)
    ls = _liquidity_score(df)

    raw = ts + ps + ls  # 最大60
    if not np.isfinite(raw):
        return None

    score = float(raw / 60.0 * 100.0)
    return float(np.clip(score, 0, 100))