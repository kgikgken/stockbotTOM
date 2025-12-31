from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.util import clamp


def _swing_low(df: pd.DataFrame, lookback: int = 12) -> float:
    low = df["Low"].astype(float)
    if len(low) < 5:
        return float(low.min())
    return float(low.tail(lookback).min())


def _stock_vs_index_rs20(stock_close: pd.Series, index_close: pd.Series) -> float:
    # RS proxy: (stock 20d return - index 20d return)
    if stock_close is None or index_close is None:
        return 0.0
    if len(stock_close) < 21 or len(index_close) < 21:
        return 0.0
    s = stock_close.astype(float)
    i = index_close.astype(float)
    sret = float(s.iloc[-1] / s.iloc[-21] - 1.0)
    iret = float(i.iloc[-1] / i.iloc[-21] - 1.0)
    v = (sret - iret) * 100.0
    if not np.isfinite(v):
        return 0.0
    return float(v)


def _pwin_proxy(setup_type: str, feat: dict, rs20: float) -> float:
    """
    Pwin proxy 0.25-0.55
    Uses:
      - trend strength (ma structure + slope)
      - RSI in sweet spot
      - liquidity (adv20)
      - volume quality (for B)
      - RS20
      - GU risk
    """
    c = feat["close"]
    ma20 = feat["ma20"]
    ma50 = feat["ma50"]
    slope = feat["ma20_slope5"]
    rsi = feat["rsi"]
    adv20 = feat["adv20"]

    base = 0.30

    # trend
    if c > ma20 > ma50:
        base += 0.06
    if slope > 0:
        base += clamp(slope * 4.0, 0.0, 0.06)

    # RSI sweet
    if 42 <= rsi <= 62:
        base += 0.05
    elif rsi < 35:
        base -= 0.04
    elif rsi > 75:
        base -= 0.03

    # RS20
    base += clamp(rs20 / 100.0, -0.04, 0.06)

    # liquidity (adv20 in JPY)
    if np.isfinite(adv20):
        if adv20 >= 1_000_000_000:
            base += 0.05
        elif adv20 >= 200_000_000:
            base += 0.03
        elif adv20 >= 100_000_000:
            base += 0.01
        else:
            base -= 0.04

    # setup bonus
    if setup_type == "A":
        base += 0.02
    else:
        base += 0.00

    return float(clamp(base, 0.25, 0.55))


def compute_rr_ev(
    hist: pd.DataFrame,
    feat: dict,
    setup_type: str,
    entry_zone: dict,
    market_ctx: dict,
    index_close: pd.Series | None,
) -> Dict:
    """
    STOP/TP:
      - stop uses structure + ATR
      - TP1/TP2 uses R multiple but capped by resistance (HH60)
    RR not fixed: TP2 capped -> RR changes.
    ExpectedDays and R/day (priority)
    EV and AdjEV (regime multiplier)
    """
    df = hist.copy()
    close = df["Close"].astype(float)
    c = float(feat["close"])
    atr = float(feat["atr"])
    if not np.isfinite(atr) or atr <= 0:
        atr = max(c * 0.01, 1.0)

    entry = float(entry_zone["center"])
    in_low = float(entry_zone["in_low"])

    swing_low = _swing_low(df, lookback=12)

    # STOP
    if setup_type == "A":
        stop = in_low - 0.7 * atr
    else:
        stop = entry - 1.0 * atr
    # structural protection
    stop = min(stop, swing_low - 0.2 * atr)

    # clamp stop distance
    stop_dist = entry - stop
    min_dist = 0.02 * entry
    max_dist = 0.10 * entry
    if stop_dist < min_dist:
        stop = entry - min_dist
    if stop_dist > max_dist:
        stop = entry - max_dist

    stop_dist = entry - stop

    # Base R plan
    r = stop_dist
    tp1 = entry + 1.5 * r
    tp2 = entry + 3.0 * r

    # Cap TP2 by resistance (HH60)
    hh60 = float(feat["hh60"]) if np.isfinite(feat["hh60"]) else float(df["High"].astype(float).max())
    cap = hh60 * 0.995
    # Also cap insane TP
    cap2 = entry * 1.25
    tp2 = min(tp2, cap, cap2)

    # Recompute RR
    rr = (tp2 - entry) / (entry - stop) if (entry - stop) > 0 else 0.0
    rr = float(rr) if np.isfinite(rr) else 0.0

    # ExpectedDays (market adaptive k)
    mkt_score = int(market_ctx["score"])
    k = 1.0
    if mkt_score >= 70:
        k = 1.15
    elif mkt_score >= 60:
        k = 1.05
    elif mkt_score >= 50:
        k = 1.00
    else:
        k = 0.90

    expected_days = (tp2 - entry) / (k * atr) if atr > 0 else 99.0
    expected_days = float(clamp(expected_days, 1.0, 10.0))

    r_per_day = rr / expected_days if expected_days > 0 else 0.0

    # Pwin proxy
    rs20 = 0.0
    if index_close is not None and len(index_close) > 0:
        rs20 = _stock_vs_index_rs20(close, index_close)
    pwin = _pwin_proxy(setup_type=setup_type, feat=feat, rs20=rs20)

    ev = pwin * rr - (1.0 - pwin) * 1.0
    reg_mult = float(market_ctx.get("regime_multiplier", 1.0))
    adj_ev = ev * reg_mult

    return {
        "entry": float(round(entry, 1)),
        "in_low": float(round(entry_zone["in_low"], 1)),
        "in_high": float(round(entry_zone["in_high"], 1)),
        "stop": float(round(stop, 1)),
        "tp1": float(round(tp1, 1)),
        "tp2": float(round(tp2, 1)),
        "rr": float(rr),
        "expected_days": float(expected_days),
        "r_per_day": float(r_per_day),
        "pwin": float(pwin),
        "ev": float(ev),
        "adj_ev": float(adj_ev),
        "atr": float(round(atr, 1)),
        "rs20": float(rs20),
    }