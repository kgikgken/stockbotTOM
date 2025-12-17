　
from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()
    df["rsi14"] = _rsi14(c)

    ret = c.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    df["turnover"] = c * v
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    df["ma20_slope5"] = df["ma20"].diff(5)
    return df


def calc_in_rank(df: pd.DataFrame) -> str:
    rsi = _last_val(df["rsi14"])
    c = _last_val(df["Close"])
    ma20 = _last_val(df["ma20"])
    if not (np.isfinite(rsi) and np.isfinite(c) and np.isfinite(ma20)):
        return "様子見"

    if 35 <= rsi <= 50 and c >= ma20 * 0.97:
        return "強IN"
    if 40 <= rsi <= 62 and c >= ma20 * 0.99:
        return "通常IN"
    if 62 < rsi <= 70:
        return "弱めIN"
    return "様子見"


def score_stock(df: pd.DataFrame) -> float | None:
    """
    0-100（順張り専用スコア）
    """
    if df is None or len(df) < 120:
        return None

    c = df["Close"].astype(float)
    ma20 = df["ma20"]
    ma50 = df["ma50"]
    ma200 = df["ma200"]
    slope5 = df["ma20_slope5"]
    t20 = df["turnover_avg20"]
    vola20 = df["vola20"]
    rsi = df["rsi14"]

    c_last = _last_val(c)
    ma20_last = _last_val(ma20)
    ma50_last = _last_val(ma50)
    ma200_last = _last_val(ma200)
    slope_last = _last_val(slope5)
    t_last = _last_val(t20)
    vola_last = _last_val(vola20)
    rsi_last = _last_val(rsi)

    sc = 0.0

    # Trend quality (max 55)
    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last > ma50_last:
            sc += 35
        elif c_last > ma20_last:
            sc += 18
        elif ma20_last > ma50_last:
            sc += 10

    if np.isfinite(ma200_last) and np.isfinite(ma50_last) and np.isfinite(c_last):
        if c_last > ma200_last and ma50_last > ma200_last:
            sc += 12
        elif c_last > ma200_last:
            sc += 6

    if np.isfinite(slope_last) and slope_last > 0:
        sc += 8

    # Pullback quality (max 25)
    if np.isfinite(rsi_last):
        if 38 <= rsi_last <= 55:
            sc += 18
        elif 55 < rsi_last <= 62:
            sc += 10
        elif 30 <= rsi_last < 38:
            sc += 10
        else:
            sc += 2

    # Liquidity + vola (max 20)
    if np.isfinite(t_last):
        if t_last >= 1e9:
            sc += 14
        elif t_last >= 1e8:
            sc += 14 * (t_last - 1e8) / 9e8

    if np.isfinite(vola_last):
        if vola_last < 0.02:
            sc += 6
        elif vola_last < 0.06:
            sc += 6 * (0.06 - vola_last) / 0.04

    return float(np.clip(sc, 0, 100))


def trend_strength(df: pd.DataFrame) -> float:
    """
    0-100: TrendGate通過後の“走行”強さ（AL3用）
    """
    c = df["Close"].astype(float)
    ma20 = df["ma20"]
    ma50 = df["ma50"]
    ma200 = df["ma200"]
    slope = df["ma20_slope5"]

    c_last = _last_val(c)
    ma20_last = _last_val(ma20)
    ma50_last = _last_val(ma50)
    ma200_last = _last_val(ma200)
    slope_last = _last_val(slope)

    s = 0.0
    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last > ma50_last:
            s += 50
    if np.isfinite(ma200_last) and np.isfinite(ma50_last) and ma50_last > ma200_last:
        s += 20
    if np.isfinite(slope_last) and slope_last > 0:
        s += 15
    if np.isfinite(c_last) and np.isfinite(ma20_last) and ma20_last > 0:
        dist = abs(c_last / ma20_last - 1.0)
        s += float(np.clip(15 * (0.04 - dist) / 0.04, 0, 15))
    return float(np.clip(s, 0, 100))
