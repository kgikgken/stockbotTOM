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
    v = df["Volume"].astype(float) if "Volume" in df.columns else 0

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

# ------------------------------------------------------------
# TrendGate（順張り専用）
# ------------------------------------------------------------
def trend_gate(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 60:
        return False

    df = _add(hist)
    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])

    if not (ma20 > ma50):
        return False
    if not (c > ma50):
        return False

    return True

# ------------------------------------------------------------
# 押し目判定（STEP2）
# ------------------------------------------------------------
def calc_inout_for_stock(hist: pd.DataFrame):
    if hist is None or len(hist) < 80:
        return "様子見", 0, 0

    df = _add(hist)

    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    rsi = _last(df["rsi"])
    off_high = _last(df["off_high"])
    turnover = _last(df["turnover"])

    # --- A. 正統派トレンド押し目 ---
    if (
        ma20 > ma50 and
        abs(c / ma20 - 1) <= 0.01 and
        40 <= rsi <= 55
    ):
        return "強IN", 0, 0

    # --- B. ブレイク後初押し ---
    if (
        off_high <= -3 and off_high >= -7 and
        ma20 > ma50 and
        turnover >= 1e8
    ):
        return "通常IN", 0, 0

    return "様子見", 0, 0


# ------------------------------------------------------------
# スコア（選別用にレンジを出す）
# ------------------------------------------------------------
def score_stock(hist: pd.DataFrame) -> float | None:
    if hist is None or len(hist) < 80:
        return None

    df = _add(hist)
    score = 0.0

    if _last(df["ma20"]) > _last(df["ma50"]):
        score += 30
    if _last(df["turnover"]) >= 1e8:
        score += 30
    if -15 <= _last(df["off_high"]) <= -3:
        score += 20
    if 40 <= _last(df["rsi"]) <= 60:
        score += 20

    return float(np.clip(score, 0, 100))