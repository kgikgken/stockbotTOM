from __future__ import annotations
import numpy as np
import pandas as pd

def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return np.nan

def _add(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else 0.0

    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma60h"] = c.rolling(60).max()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["turnover"] = (c * v).rolling(20).mean()
    df["off_high"] = (c - df["ma60h"]) / df["ma60h"] * 100
    return df


# ============================================================
# 週足 TrendGate（最重要）
# ============================================================
def trend_gate_weekly(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 100:
        return False

    w = hist.resample("W").last()
    if len(w) < 60:
        return False

    c = w["Close"].astype(float)
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()

    return _last(ma20) > _last(ma50) and _last(c) > _last(ma50)


# ============================================================
# 日足 TrendGate
# ============================================================
def trend_gate_daily(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 60:
        return False

    df = _add(hist)
    return _last(df["ma20"]) > _last(df["ma50"]) and _last(df["Close"]) > _last(df["ma50"])


# ============================================================
# 押し目判定
# ============================================================
def calc_inout_for_stock(hist: pd.DataFrame):
    if hist is None or len(hist) < 80:
        return "様子見", 0, 0

    df = _add(hist)
    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    rsi = _last(df["rsi"])
    off = _last(df["off_high"])
    t = _last(df["turnover"])

    # 正統派トレンド押し目
    if ma20 > ma50 and abs(c / ma20 - 1) <= 0.01 and 40 <= rsi <= 55:
        return "強IN", 0, 0

    # ブレイク後初押し
    if ma20 > ma50 and -7 <= off <= -3 and t >= 1e8:
        return "通常IN", 0, 0

    return "様子見", 0, 0