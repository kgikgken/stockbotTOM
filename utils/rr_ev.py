from __future__ import annotations

import numpy as np

from utils.util import clamp


def rr(entry: float, sl: float, tp: float) -> float:
    risk = entry - sl
    reward = tp - entry
    if risk <= 0:
        return float("nan")
    return float(reward / risk)


def expected_days(entry: float, tp1: float, atr: float) -> float:
    """Mechanical estimate of holding days.

    v2.3+ spec: Expected days must be derived mechanically.
    We use the TP1 distance in ATR units as a base estimator.
    This is later clipped and penalized by the caller.
    """
    if atr <= 0:
        return float("nan")
    return float((tp1 - entry) / atr)


def expected_value(tp1_rr: float, reach_prob: float) -> float:
    """Expected value in R-units using TP1 RR and reach probability.

    Assumes -1R loss when stop hits.
    EV = p*RR - (1-p)
    """
    p = max(0.0, min(1.0, float(reach_prob)))
    return float(p * tp1_rr - (1.0 - p))


def turnover_efficiency(tp1_rr: float, exp_days: float) -> float:
    """R per day proxy: tp1 RR divided by expected days."""
    d = max(1e-9, float(exp_days))
    return float(tp1_rr / d)


def cagr_contribution(tp1_rr: float, reach_prob: float, exp_days: float) -> float:
    """Core ranking score: (expectedR * reachProb) / expectedDays."""
    d = max(1e-9, float(exp_days))
    p = max(0.0, min(1.0, float(reach_prob)))
    return float((tp1_rr * p) / d)


def adj_ev(raw_ev: float, market_score: int, macro_caution: bool, risk_on: bool) -> float:
    """Expected value after risk overlays.

    v2.3+ spec: MarketScore is *not* a selection gate. It is used primarily
    for withdrawal speed control. Here we only apply a light overlay for
    *macro* caution. The market_score argument is kept for compatibility.
    """
    x = float(raw_ev)

    if macro_caution and not risk_on:
        x *= 0.90
    elif macro_caution and risk_on:
        x *= 0.95

    return float(clamp(x, -2.0, 3.0))
