from __future__ import annotations
import numpy as np
import pandas as pd


# =============================
# Helpers
# =============================
def _last(series: pd.Series) -> float:
    try:
        v = float(series.iloc[-1])
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return np.nan


def _vola20(close: pd.Series) -> float:
    ret = close.pct_change(fill_method=None)
    v = ret.rolling(20).std().iloc[-1]
    try:
        return float(v)
    except Exception:
        return 0.03


# =============================
# Base TP/SL
# =============================
def base_tp_sl(vola20: float):
    if vola20 < 0.015:
        return 0.06, -0.03
    if vola20 < 0.03:
        return 0.08, -0.04
    return 0.12, -0.06


# =============================
# Market adjust
# =============================
def mkt_adjust(tp: float, sl: float, mkt_score: int):
    if mkt_score >= 70:
        tp += 0.02
        sl = min(sl, -0.03)
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)
    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))
    return tp, sl


# =============================
# Entry price
# =============================
def compute_entry(hist: pd.DataFrame) -> float:
    close = hist["Close"].astype(float)
    price = _last(close)

    # MA
    ma20 = close.rolling(20).mean().iloc[-1]
    if not np.isfinite(ma20):
        return price

    # ATR
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    # 基本: MA20 - ATR*0.4（深押しを待つ）
    entry = ma20
    if np.isfinite(atr):
        entry = ma20 - atr * 0.40

    # 現値より上にならない
    if entry > price:
        entry = price * 0.995

    # 深すぎ防止: 過去5日安値割らない
    last_low = float(low.iloc[-5:].min())
    if entry < last_low:
        entry = last_low * 1.02

    return float(entry)


# =============================
# Pullback coefficient
# =============================
def pullback_coef(hist: pd.DataFrame) -> float:
    close = hist["Close"]
    rsi = _rsi(close)
    off = (close.iloc[-1] / close.rolling(60).max().iloc[-1] - 1.0) * 100.0

    c = 1.0

    if np.isfinite(rsi):
        if 32 <= rsi <= 45:
            c *= 1.10
        elif 26 <= rsi < 32:
            c *= 1.05

    if np.isfinite(off):
        if -15 <= off <= -5:
            c *= 1.10
        elif -22 <= off < -15:
            c *= 1.05

    return float(c)


def _rsi(close: pd.Series) -> float:
    if len(close) < 16:
        return 50.0
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    r = 100 - (100 / (1 + rs))
    v = r.iloc[-1]
    try:
        return float(v)
    except Exception:
        return 50.0


# =============================
# Main
# =============================
def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> dict:
    close = hist["Close"].astype(float)
    price = _last(close)

    # vola
    vola = _vola20(close)

    # base
    tp, sl = base_tp_sl(vola)

    # pullback
    coef = pullback_coef(hist)
    tp *= coef

    # mkt adjust
    tp, sl = mkt_adjust(tp, sl, mkt_score)

    # entry
    entry = compute_entry(hist)
    if not np.isfinite(entry):
        entry = price

    # RR
    rr = abs(tp / abs(sl)) if sl != 0 else 0.0

    return {
        "entry": entry,
        "tp_pct": tp,
        "sl_pct": sl,
        "rr": float(rr)
    }