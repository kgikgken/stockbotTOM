from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return float('nan')


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 0.0

    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    v = tr.rolling(period).mean().iloc[-1]
    if v is None or not np.isfinite(v):
        return 0.0
    return float(v)


def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int, for_day: bool = False) -> dict:
    df = hist.copy()
    close = df['Close'].astype(float)
    price = _last_val(close)
    if not np.isfinite(price) or price <= 0:
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0, tp_price=0.0, sl_price=0.0, entry_basis='na')

    atr = _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(price * 0.01, 1.0)

    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
    ma5 = float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else price

    entry = ma20 - 0.5 * atr
    entry_basis = 'pullback'

    if price > ma5 > ma20:
        entry = ma20 + (ma5 - ma20) * 0.25
        entry_basis = 'trend_pullback'

    if entry > price:
        entry = price * 0.995

    lookback = 8 if for_day else 12
    swing_low = float(df['Low'].astype(float).tail(lookback).min())
    sl_price = min(entry - 0.8 * atr, swing_low - 0.2 * atr)

    sl_pct = (sl_price / entry - 1.0)
    sl_pct = float(np.clip(sl_pct, -0.10, -0.02))
    sl_price = entry * (1.0 + sl_pct)

    hi_window = 60 if len(close) >= 60 else len(close)
    high_60 = float(close.tail(hi_window).max())

    tp_price = min(high_60 * 0.995, entry * (1.0 + (0.22 if not for_day else 0.08)))

    if mkt_score >= 70:
        tp_price *= 1.03
    elif mkt_score <= 45:
        tp_price *= 0.97

    if tp_price <= entry:
        tp_price = entry * (1.0 + (0.06 if not for_day else 0.03))

    tp_pct = (tp_price / entry - 1.0)
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
