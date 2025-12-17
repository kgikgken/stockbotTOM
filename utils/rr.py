
from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 0.0
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else 0.0


def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> dict:
    """
    可変RR（固定禁止）
    - entry: MA20 - 0.5ATR（ただし現値より上にしない）
    - SL: 構造（直近安値）+ ATR をベースに、%クランプ
    - TP: 60日高値手前 or +22% 上限（抵抗に届く範囲）
    """
    df = hist.copy()
    close = df["Close"].astype(float)
    price = _last_val(close)
    if not np.isfinite(price) or price <= 0:
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0, tp_price=0.0, sl_price=0.0, entry_basis="na")

    atr = _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(price * 0.01, 1.0)

    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
    entry = ma20 - 0.5 * atr
    entry_basis = "pullback_ma20"

    if entry > price:
        entry = price * 0.995

    swing_low = float(df["Low"].astype(float).tail(12).min())
    sl_price = min(entry - 0.8 * atr, swing_low - 0.2 * atr)

    sl_pct = float((sl_price / entry) - 1.0)
    sl_pct = float(np.clip(sl_pct, -0.10, -0.02))
    sl_price = entry * (1.0 + sl_pct)

    hi_window = 60 if len(close) >= 60 else len(close)
    high_60 = float(close.tail(hi_window).max())
    tp_price = min(high_60 * 0.995, entry * 1.22)

    if mkt_score >= 70:
        tp_price *= 1.03
    elif mkt_score <= 45:
        tp_price *= 0.97

    if tp_price <= entry:
        tp_price = entry * 1.06

    tp_pct = float((tp_price / entry) - 1.0)
    tp_pct = float(np.clip(tp_pct, 0.03, 0.30))
    tp_price = entry * (1.0 + tp_pct)

    rr = tp_pct / abs(sl_pct) if sl_pct < 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(round(entry, 1)),
        tp_pct=float(tp_pct),
        sl_pct=float(sl_pct),
        tp_price=float(round(tp_price, 1)),
        sl_price=float(round(sl_price, 1)),
        entry_basis=entry_basis,
    )
