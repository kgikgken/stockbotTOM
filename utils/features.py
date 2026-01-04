from __future__ import annotations

import numpy as np
import pandas as pd

from utils.util import clamp


def _last(x: pd.Series) -> float:
    try:
        return float(x.iloc[-1])
    except Exception:
        return np.nan


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else np.nan


def compute_indicators(hist: pd.DataFrame) -> dict:
    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    vol = hist["Volume"].astype(float) if "Volume" in hist.columns else pd.Series(np.nan, index=hist.index)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma10 = close.rolling(10).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))

    atr = _atr(hist, 14)
    c = _last(close)

    # 売買代金（JPY換算として proxy: close*volume）
    turnover = close * vol
    adv20 = _last(turnover.rolling(20).mean())

    # 20日高値（ブレイク判定）
    hh20 = _last(close.rolling(20).max())

    return {
        "close": c,
        "ma10": _last(ma10),
        "ma20": _last(ma20),
        "ma50": _last(ma50),
        "rsi14": _last(rsi14),
        "atr": float(atr) if np.isfinite(atr) else np.nan,
        "adv20": float(adv20) if np.isfinite(adv20) else np.nan,
        "hh20": float(hh20) if np.isfinite(hh20) else np.nan,
        "high": float(_last(high)),
        "low": float(_last(low)),
    }


def atr_percent(ind: dict) -> float:
    c = float(ind.get("close", np.nan))
    atr = float(ind.get("atr", np.nan))
    if not (np.isfinite(c) and c > 0 and np.isfinite(atr) and atr > 0):
        return 0.0
    return float(atr / c * 100.0)


def setup_type(ind: dict) -> str:
    """
    Setup A/Bのみ（逆張りOFF）
    """
    c = ind["close"]
    ma20 = ind["ma20"]
    ma50 = ind["ma50"]
    rsi = ind["rsi14"]
    hh20 = ind["hh20"]

    if not all(np.isfinite([c, ma20, ma50, rsi])):
        return "-"

    # A: トレンド押し目
    if c > ma20 > ma50 and 40 <= rsi <= 62:
        return "A"

    # B: ブレイク（厳選。追いかけ禁止なのでエントリーは後で帯チェック）
    if np.isfinite(hh20) and c > hh20 and c > ma20 > ma50:
        return "B"

    return "-"