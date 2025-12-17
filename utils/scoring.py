from __future__ import annotations

import numpy as np
import pandas as pd

def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan

def _sma(series: pd.Series, n: int) -> float:
    if series is None or len(series) < n:
        return _last_val(series)
    return float(series.rolling(n).mean().iloc[-1])

def _add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / (rolling_high + 1e-9) * 100
    else:
        df["off_high_pct"] = np.nan

    df["trend_slope20"] = df["ma20"].pct_change(fill_method=None)

    rng = (high - low).replace(0, np.nan)
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(np.isfinite(rng), lower_shadow / rng, 0.0)

    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()
    return df

def score_stock(hist: pd.DataFrame) -> float | None:
    """0-100 base score (trend + pullback quality + liquidity)."""
    if hist is None or len(hist) < 120:
        return None
    df = _add_indicators(hist)

    close = df["Close"].astype(float)
    c_last = _last_val(close)
    ma20_last = _last_val(df["ma20"])
    ma50_last = _last_val(df["ma50"])
    slope_last = _last_val(df["trend_slope20"])

    rsi = _last_val(df["rsi14"])
    off = _last_val(df["off_high_pct"])
    shadow = _last_val(df["lower_shadow_ratio"])
    t = _last_val(df["turnover_avg20"])
    vola = _last_val(df["vola20"])

    sc = 0.0

    # Trend (max 35)
    if np.isfinite(slope_last):
        if slope_last >= 0.01:
            sc += 14
        elif slope_last > 0:
            sc += 7 + (slope_last / 0.01) * 7
        else:
            sc += max(0.0, 7 + slope_last * 70)

    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last > ma50_last:
            sc += 14
        elif c_last > ma20_last:
            sc += 7
        elif ma20_last > ma50_last:
            sc += 3

    # Pullback / not overheated (max 30)
    if np.isfinite(rsi):
        if 30 <= rsi <= 50:
            sc += 14
        elif 50 < rsi <= 62:
            sc += 10
        elif 25 <= rsi < 30 or 62 < rsi <= 70:
            sc += 6
        else:
            sc += 2

    if np.isfinite(off):
        if -18 <= off <= -5:
            sc += 10
        elif -25 <= off < -18:
            sc += 6
        elif -5 < off <= 5:
            sc += 6
        else:
            sc += 2

    if np.isfinite(shadow):
        if shadow >= 0.5:
            sc += 6
        elif shadow >= 0.3:
            sc += 3

    # Liquidity / volatility comfort (max 35)
    if np.isfinite(t):
        if t >= 1e9:
            sc += 25
        elif t >= 1e8:
            sc += 25 * (t - 1e8) / 9e8

    if np.isfinite(vola):
        if vola < 0.02:
            sc += 10
        elif vola < 0.06:
            sc += 10 * (0.06 - vola) / 0.04

    return float(np.clip(sc, 0, 100))

def calc_in_rank(hist: pd.DataFrame) -> str:
    """Return 強IN/通常IN/弱めIN/様子見 based on pullback condition."""
    if hist is None or len(hist) < 120:
        return "様子見"
    df = _add_indicators(hist)

    close_last = _last_val(df["Close"])
    ma20_last = _last_val(df["ma20"])
    rsi_last = _last_val(df["rsi14"])
    off_last = _last_val(df["off_high_pct"])
    shadow = _last_val(df["lower_shadow_ratio"])

    if not (np.isfinite(rsi_last) and np.isfinite(ma20_last) and np.isfinite(close_last) and np.isfinite(off_last)):
        return "様子見"

    # Strong pullback completion-ish
    if 30 <= rsi_last <= 50 and -20 <= off_last <= -5 and close_last >= ma20_last * 0.97 and shadow >= 0.3:
        return "強IN"
    # Normal
    if 40 <= rsi_last <= 64 and -15 <= off_last <= 5 and close_last >= ma20_last * 0.99:
        return "通常IN"
    # Weak
    if 25 <= rsi_last < 30 or 64 < rsi_last <= 72:
        return "弱めIN"

    return "様子見"

def trend_gate(hist: pd.DataFrame) -> tuple[bool, dict]:
    """Exclude 'falling knife pullback' (逆張り) by requiring short-term uptrend structure.
    Returns (ok, details)
    """
    if hist is None or len(hist) < 220:
        return False, {"reason": "data_short"}

    close = hist["Close"].astype(float)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    c = _last_val(close)
    m20 = _last_val(ma20)
    m50 = _last_val(ma50)
    m200 = _last_val(ma200)

    if not all(np.isfinite(x) for x in (c, m20, m50, m200)):
        return False, {"reason": "nan"}

    # Structure: MA20 > MA50 > MA200 and price above MA50
    struct_ok = (m20 > m50 > m200) and (c > m50)

    # Rising MA50 slope (20 trading days)
    if len(ma50.dropna()) < 25:
        slope_ok = False
    else:
        m50_now = float(ma50.dropna().iloc[-1])
        m50_prev = float(ma50.dropna().iloc[-21])
        slope_ok = (m50_now / m50_prev - 1.0) > 0.0

    ok = bool(struct_ok and slope_ok)
    return ok, {
        "price": float(c),
        "ma20": float(m20),
        "ma50": float(m50),
        "ma200": float(m200),
        "struct_ok": bool(struct_ok),
        "ma50_slope_ok": bool(slope_ok),
    }
