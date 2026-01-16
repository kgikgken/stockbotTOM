from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.util import clamp, PriceBand


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float("nan")


def _swing_low(df: pd.DataFrame, lookback: int = 12) -> float:
    try:
        return float(df["Low"].astype(float).tail(lookback).min())
    except Exception:
        return float("nan")


def build_entry_band(setup: str, sma20: float, atr: float, breakout_line: float | None = None) -> PriceBand:
    """Entry band definition (internal). Display is center only.

    - A1/A1-Strong/A2: SMA20 ± 0.5ATR
    - B: breakout line ± 0.3ATR

    Caps to avoid overly wide bands:
      - cap_width <= 1.2ATR (A) / 0.9ATR (B)
      - cap_width_pct <= 8% of center price
    """
    if not np.isfinite(atr) or atr <= 0:
        atr = max(1.0, abs(sma20) * 0.01)

    if setup == "B" and breakout_line is not None and np.isfinite(float(breakout_line)):
        center = float(breakout_line)
        half = 0.3 * atr
        cap_atr = 0.45 * atr
    else:
        center = float(sma20) if np.isfinite(sma20) else float(breakout_line or 0.0)
        half = 0.5 * atr
        cap_atr = 0.60 * atr

    # cap by ATR
    half = float(min(half, cap_atr))

    # cap by %
    cap_pct = abs(center) * 0.04  # 8% total width
    half = float(min(half, cap_pct))

    low = center - half
    high = center + half
    return PriceBand(low=float(low), high=float(high))


def compute_trade_plan(
    df: pd.DataFrame,
    setup: str,
    atr: float,
    sma20: float,
    mkt_score: int,
    macro_caution: bool,
    allow_tp2_tight: bool,
) -> Dict:
    """Compute SL/TP and metrics.

    Exit design:
      - SL: entry_center - 1.2ATR, but use recent swing low if closer (priority)
      - TP1: +1.5R
      - TP2: variable 2.0~3.5R (tighten under macro caution)

    Metrics:
      - RR = (TP2-entry)/(entry-SL)
      - ExpectedDays = (TP2-entry)/ATR
      - R/day = RR / ExpectedDays

    EV display is normalized AdjEV in [ -0.50, 1.20 ] with threshold 0.50.
    """
    close = df["Close"].astype(float)
    price = _last(close)

    # breakout line for B
    hh20 = float(close.rolling(20).max().iloc[-1]) if len(close) >= 20 else float(price)

    band = build_entry_band(setup=setup, sma20=sma20, atr=atr, breakout_line=hh20)
    entry = float(band.center())

    # SL
    base_sl = entry - 1.2 * atr
    swing_low = _swing_low(df, lookback=12)
    if np.isfinite(swing_low):
        # prefer structure if it is closer (less risk) while still below entry
        struct_sl = min(swing_low, entry - 0.2 * atr)
        sl = max(base_sl, struct_sl)  # closer to entry
    else:
        sl = base_sl

    # enforce min/max risk
    risk = entry - sl
    if not np.isfinite(risk) or risk <= 0:
        sl = entry - max(atr, entry * 0.02)
        risk = entry - sl

    # RR lower bound by mkt score
    if mkt_score >= 70:
        rr_min = 1.8
    elif mkt_score >= 60:
        rr_min = 2.0
    elif mkt_score >= 50:
        rr_min = 2.2
    else:
        rr_min = 2.5

    # TP2 target RR (2.0..3.5). Keep stable, avoid excessive freedom.
    base_rr = 2.6 if mkt_score >= 60 else 2.4
    if setup == "A1-Strong":
        base_rr += 0.05
    elif setup == "B":
        base_rr += 0.15

    # Macro caution tightens TP2 a bit
    if allow_tp2_tight:
        base_rr *= 0.92

    rr_target = float(clamp(base_rr, 2.0, 3.5))
    rr_target = max(rr_target, rr_min)

    tp2 = entry + risk * rr_target
    tp1 = entry + risk * 1.5

    rr = (tp2 - entry) / risk

    # expected days
    expected_days = (tp2 - entry) / atr if np.isfinite(atr) and atr > 0 else float("nan")
    expected_days = float(clamp(expected_days, 1.0, 8.0)) if np.isfinite(expected_days) else 5.0

    rday = float(rr / expected_days) if expected_days > 0 else 0.0

    # Setup-dependent R/day floor (v2.3)
    if setup == "A1-Strong":
        rday_min = 0.45
    elif setup == "A1":
        rday_min = 0.45
    elif setup == "A2":
        rday_min = 0.50
    else:
        rday_min = 0.65

    # Structural EV (compressed 3-factor) -> normalized AdjEV
    # Raw: RR * TrendStrength * PullbackQuality is handled outside; here we accept already-aggregated value.
    # We still produce a stable surrogate by using RR and rday only (prevents black-box explosion).
    # NOTE: This keeps thresholds consistent and avoids runaway AdjEV.
    ev_raw = float(rr * (0.5 + clamp(rday, 0.0, 1.5) / 3.0))  # 1.0..~2.3
    adjev = float(clamp(ev_raw * 0.35, -0.50, 1.20))

    action = "指値（Entryで待つ）"
    if macro_caution:
        action = "指値（ロット50%・TP2控えめ）" if allow_tp2_tight else "指値（ロット50%）"

    return {
        "entry_band_low": float(band.low),
        "entry_band_high": float(band.high),
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "rr_min": float(rr_min),
        "expected_days": float(expected_days),
        "rday": float(rday),
        "rday_min": float(rday_min),
        "adjev": float(adjev),
        "action": action,
        "price_now": float(price) if np.isfinite(price) else float("nan"),
    }
