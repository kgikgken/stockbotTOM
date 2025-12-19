from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return np.nan


def _add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    o = df["Open"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else 0

    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["rsi"] = 100 - 100 / (1 + c.diff().clip(lower=0).rolling(14).mean() /
                             (c.diff().clip(upper=0).abs().rolling(14).mean() + 1e-9))
    df["off_high"] = (c - c.rolling(60).max()) / c.rolling(60).max() * 100
    df["turnover"] = (c * v).rolling(20).mean()

    return df


# ============================================================
# TrendGate（逆張り完全排除）
# ============================================================
def trend_gate(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 60:
        return False

    df = _add_indicators(hist)

    c = _last_val(df["Close"])
    ma20 = _last_val(df["ma20"])
    ma50 = _last_val(df["ma50"])

    try:
        ma50_prev = float(df["ma50"].iloc[-6])
        slope_ok = ma50 > ma50_prev
    except Exception:
        slope_ok = False

    if not np.isfinite(c) or not np.isfinite(ma20) or not np.isfinite(ma50):
        return False
    if not (ma20 > ma50):
        return False
    if not slope_ok:
        return False
    if not (c > ma50):
        return False

    return True


def score_stock(hist: pd.DataFrame) -> float | None:
    if hist is None or len(hist) < 80:
        return None

    df = _add_indicators(hist)

    sc = 0.0
    if _last_val(df["ma20"]) > _last_val(df["ma50"]):
        sc += 30
    if _last_val(df["rsi"]) >= 40:
        sc += 20
    if _last_val(df["off_high"]) >= -15:
        sc += 20
    if _last_val(df["turnover"]) >= 1e8:
        sc += 30

    return float(np.clip(sc, 0, 100))


def calc_inout_for_stock(hist: pd.DataFrame):
    df = _add_indicators(hist)
    rsi = _last_val(df["rsi"])
    off = _last_val(df["off_high"])

    if 40 <= rsi <= 55 and -12 <= off <= -3:
        return "強IN", 0, 0
    if 45 <= rsi <= 62 and -10 <= off <= 5:
        return "通常IN", 0, 0
    if 35 <= rsi < 40:
        return "弱めIN", 0, 0
    return "様子見", 0, 0