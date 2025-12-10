from __future__ import annotations
import numpy as np
import pandas as pd


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    # MA
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ma100"] = close.rolling(100).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 60日高値からの位置
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
    else:
        df["off_high_pct"] = np.nan

    # 20日ボラ
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # 出来高・売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


def score_stock(ticker: str, hist: pd.DataFrame, uni_row) -> float | None:
    """
    0〜100 の Quality スコア
    - トレンド
    - 押し目位置
    - 流動性
    """
    if hist is None or len(hist) < 60:
        return None

    df = _add_indicators(hist)

    score = 0.0

    # トレンド
    ma20 = df["ma20"]
    ma50 = df["ma50"]
    ma100 = df["ma100"]

    c_last = _last(df["Close"])
    ma20_last = _last(ma20)
    ma50_last = _last(ma50)
    ma100_last = _last(ma100)

    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last > ma50_last:
            score += 18
        elif c_last > ma20_last and ma20_last > ma100_last:
            score += 14
        elif ma20_last > ma50_last:
            score += 8
        else:
            score += 3

    # RSI
    rsi = _last(df["rsi14"])
    if np.isfinite(rsi):
        if 35 <= rsi <= 60:
            score += 14
        elif 30 <= rsi < 35 or 60 < rsi <= 65:
            score += 7
        else:
            score += 2

    # 高値からの押し目
    off = _last(df["off_high_pct"])
    if np.isfinite(off):
        if -18 <= off <= -5:
            score += 16
        elif -30 <= off < -18:
            score += 10
        elif -5 < off <= 5:
            score += 6
        else:
            score += 2

    # 流動性
    t20 = _last(df["turnover_avg20"])
    if np.isfinite(t20):
        if t20 >= 5e8:
            score += 18
        elif t20 >= 1e8:
            score += 10 + 8 * (t20 - 1e8) / (4e8)
        elif t20 >= 5e7:
            score += 6
        else:
            score += 1

    # ボラ（極端すぎないこと）
    v20 = _last(df["vola20"])
    if np.isfinite(v20):
        if 0.015 <= v20 <= 0.06:
            score += 14
        elif 0.01 <= v20 < 0.015 or 0.06 < v20 <= 0.08:
            score += 8
        else:
            score += 3

    score = float(np.clip(score, 0, 100))
    return score