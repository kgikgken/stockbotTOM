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
    df["ma60"] = c.rolling(60).mean()
    df["high60"] = c.rolling(60).max()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["turnover"] = (c * v).rolling(20).mean()
    df["off_high"] = (c / df["high60"] - 1.0) * 100
    return df

# ============================================================
# TrendGate（絶対条件：逆張り完全排除）
# ============================================================
def trend_gate(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 80:
        return False

    df = _add(hist)
    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    ma60 = _last(df["ma60"])

    if not (ma20 > ma50 > ma60):
        return False
    if not (c > ma20):
        return False

    return True

# ============================================================
# IN判定（勝てる形だけ）
# ============================================================
def calc_inout_for_stock(hist: pd.DataFrame):
    if hist is None or len(hist) < 80:
        return "様子見", 0, 0

    df = _add(hist)
    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    rsi = _last(df["rsi"])
    off = _last(df["off_high"])
    turnover = _last(df["turnover"])

    # A. トレンド押し目（主力）
    if (
        abs(c / ma20 - 1) <= 0.01 and
        45 <= rsi <= 60 and
        -10 <= off <= -3
    ):
        return "強IN", 0, 0

    # B. ブレイク後の初押し
    if (
        -7 <= off <= -3 and
        rsi >= 50 and
        turnover >= 1e8
    ):
        return "通常IN", 0, 0

    return "様子見", 0, 0

# ============================================================
# score（内部フィルタ専用：表示しない）
# ============================================================
def score_stock(hist: pd.DataFrame) -> float | None:
    if hist is None or len(hist) < 80:
        return None

    df = _add(hist)
    score = 0.0

    if _last(df["ma20"]) > _last(df["ma50"]) > _last(df["ma60"]):
        score += 40
    if _last(df["turnover"]) >= 1e8:
        score += 30
    if -12 <= _last(df["off_high"]) <= -3:
        score += 20
    if 45 <= _last(df["rsi"]) <= 65:
        score += 10

    return float(score)