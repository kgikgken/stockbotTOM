from __future__ import annotations

from typing import Tuple

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

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0
    else:
        df["off_high_pct"] = np.nan

    df["turnover"] = close * vol

    return df


def score_stock(hist: pd.DataFrame) -> float:
    if hist is None or len(hist) < 60:
        return 0.0

    df = _add_indicators(hist)
    score = 0.0

    ma20 = df["ma20"]
    slope = ma20.pct_change(fill_method=None)
    slope_last = _last(slope)

    if np.isfinite(slope_last):
        if slope_last > 0.0:
            score += min(30.0, slope_last * 2000.0 + 10.0)
        else:
            score += max(0.0, 10.0 + slope_last * 1500.0)

    rsi = _last(df["rsi14"])
    if np.isfinite(rsi):
        if 40 <= rsi <= 60:
            score += 20.0
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            score += 10.0

    off = _last(df["off_high_pct"])
    if np.isfinite(off):
        if -20 <= off <= -5:
            score += 15.0
        elif -30 <= off < -20:
            score += 8.0

    turn = _last(df["turnover"])
    if np.isfinite(turn):
        if turn >= 1e8:
            score += 20.0
        elif turn >= 5e7:
            score += 10.0

    vola = _last(df["vola20"])
    if np.isfinite(vola) and vola > 0:
        if 0.01 <= vola <= 0.04:
            score += 10.0

    return float(np.clip(score, 0.0, 100.0))


def calc_inout_for_stock(hist: pd.DataFrame) -> Tuple[str, float, float]:
    """
    INランク, TP%, SL% を決める
    戻り値: (in_rank, tp_pct, sl_pct(マイナス))
    """
    if hist is None or len(hist) < 60:
        return "様子見", 0.0, 0.0

    df = _add_indicators(hist)
    close = df["Close"].astype(float)

    price = _last(close)
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    rsi = _last(df["rsi14"])
    off = _last(df["off_high_pct"])
    vola = _last(df["vola20"])

    in_rank = "様子見"
    tp_pct = 0.0
    sl_pct = 0.0

    if not all(np.isfinite(x) for x in [price, ma20, ma50, rsi]):
        return "様子見", 0.0, 0.0

    strong_trend = price > ma20 > ma50
    up_trend = price > ma20 and ma20 > ma50 * 0.98

    if strong_trend and -20 <= off <= -5 and 40 <= rsi <= 60:
        in_rank = "強IN"
        sl_pct = -5.0
        tp_pct = 15.0
    elif up_trend and -30 <= off <= 0 and 35 <= rsi <= 65:
        in_rank = "通常IN"
        sl_pct = -6.0
        tp_pct = 14.0
    elif up_trend and -35 <= off <= 5 and 30 <= rsi <= 70:
        in_rank = "弱めIN"
        sl_pct = -7.0
        tp_pct = 12.0
    else:
        return "様子見", 0.0, 0.0

    if np.isfinite(vola) and vola > 0:
        if vola > 0.04:
            sl_pct *= 1.3
            tp_pct *= 1.1
        elif vola < 0.015:
            sl_pct *= 0.8
            tp_pct *= 0.9

    return in_rank, float(tp_pct), float(sl_pct)