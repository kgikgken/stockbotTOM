from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    rolling_high = close.rolling(60).max()
    df["off_high_pct"] = (close - rolling_high) / (rolling_high + 1e-9) * 100

    df["trend_slope20"] = df["ma20"].pct_change(fill_method=None)

    rng = (high - low).replace(0, np.nan)
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(np.isfinite(rng), lower_shadow / rng, 0.0)

    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


def score_stock(hist: pd.DataFrame) -> float | None:
    if hist is None or len(hist) < 80:
        return None

    df = _add_indicators(hist)

    sc = 0.0

    slope = _last_val(df["trend_slope20"])
    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 12
        elif slope > 0:
            sc += 6 + slope / 0.01 * 6

    c = _last_val(df["Close"])
    ma20 = _last_val(df["ma20"])
    ma50 = _last_val(df["ma50"])
    if c > ma20 > ma50:
        sc += 12
    elif c > ma20:
        sc += 6

    rsi = _last_val(df["rsi14"])
    if 30 <= rsi <= 45:
        sc += 14
    elif 45 < rsi <= 60:
        sc += 10

    off = _last_val(df["off_high_pct"])
    if -18 <= off <= -5:
        sc += 10
    elif -25 <= off < -18:
        sc += 6

    shadow = _last_val(df["lower_shadow_ratio"])
    if shadow >= 0.5:
        sc += 6
    elif shadow >= 0.3:
        sc += 3

    t = _last_val(df["turnover_avg20"])
    if t >= 1e9:
        sc += 28
    elif t >= 1e8:
        sc += 28 * (t - 1e8) / 9e8

    vola = _last_val(df["vola20"])
    if vola < 0.02:
        sc += 12
    elif vola < 0.06:
        sc += 12 * (0.06 - vola) / 0.04

    return float(np.clip(sc, 0, 100))


def calc_inout_for_stock(hist: pd.DataFrame):
    if hist is None or len(hist) < 80:
        return "様子見", 6.0, -3.0

    df = _add_indicators(hist)

    c = _last_val(df["Close"])
    ma20 = _last_val(df["ma20"])
    rsi = _last_val(df["rsi14"])
    off = _last_val(df["off_high_pct"])
    shadow = _last_val(df["lower_shadow_ratio"])

    rank = "様子見"

    if 30 <= rsi <= 48 and -20 <= off <= -5 and c >= ma20 * 0.97 and shadow >= 0.3:
        rank = "強IN"
    elif 40 <= rsi <= 62 and -15 <= off <= 5 and c >= ma20 * 0.99:
        rank = "通常IN"
    elif 25 <= rsi < 30 or 62 < rsi <= 72:
        rank = "弱めIN"

    return rank, 0.0, 0.0