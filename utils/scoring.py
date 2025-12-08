# utils/scoring.py
# ============================================================
# 銘柄スコアリング（Coreスコア: 0〜100）
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# 内部ヘルパー：インジケータ計算
# ============================================================
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    hist（yfinanceのhistory）を受け取り、
    スコアに使う指標を全部載せる。
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
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 20日ボラ（リターンの標準偏差）
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # 60日高値からの位置 & そこからの日数
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

    # 20MAの傾き
    df["trend_slope20"] = df["ma20"].pct_change(fill_method=None)

    # 下ヒゲ比率
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    # 出来高・売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


def _last_val(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    v = series.iloc[-1]
    try:
        return float(v)
    except Exception:
        return np.nan


# ============================================================
# スコア（0〜100）を計算
# ============================================================
def _trend_score(df: pd.DataFrame) -> float:
    close = df["Close"].astype(float)
    ma20 = df["ma20"]
    ma50 = df["ma50"]
    slope = df["trend_slope20"]

    sc = 0.0
    s_last = _last_val(slope)
    c_last = _last_val(close)
    ma20_last = _last_val(ma20)
    ma50_last = _last_val(ma50)

    # 傾き
    if np.isfinite(s_last):
        if s_last >= 0.01:
            sc += 8
        elif s_last > 0:
            sc += 4 + (s_last / 0.01) * 4  # 0〜0.01の間を線形
        else:
            sc += max(0.0, 4 + s_last * 50)  # -0.08でほぼ0

    # MAの並び
    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last and ma20_last > ma50_last:
            sc += 8
        elif c_last > ma20_last:
            sc += 4
        elif ma20_last > ma50_last:
            sc += 2

    # 高値からの位置（浅すぎる押しを少し評価）
    off = _last_val(df["off_high_pct"])
    if np.isfinite(off):
        if off >= -5:
            sc += 4
        elif off >= -15:
            sc += 4 - abs(off + 5) * 0.2  # -5〜-15で減点

    return float(np.clip(sc, 0, 20))


def _pullback_score(df: pd.DataFrame) -> float:
    rsi = _last_val(df["rsi14"])
    off = _last_val(df["off_high_pct"])
    days = _last_val(df["days_since_high60"])
    shadow = _last_val(df["lower_shadow_ratio"])

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


def _liquidity_score(df: pd.DataFrame) -> float:
    t = _last_val(df["turnover_avg20"])
    v = _last_val(df["vola20"])
    sc = 0.0

    # 流動性
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
    銘柄のCoreスコア（0〜100）
    Aランク: score >= 80
    Bランク: 70 <= score < 80
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

    score = float(raw / 60.0 * 100.0)  # 0〜100にスケール
    return float(np.clip(score, 0, 100))