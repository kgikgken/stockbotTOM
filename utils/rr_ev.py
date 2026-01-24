from __future__ import annotations

import numpy as np

from utils.util import clamp


def rr(entry: float, sl: float, tp: float) -> float:
    risk = entry - sl
    reward = tp - entry
    if risk <= 0:
        return float("nan")
    return float(reward / risk)


def expected_days(entry: float, tp2: float, atr: float) -> float:
    if atr <= 0:
        return float("nan")
    return float((tp2 - entry) / atr)


def turnover_efficiency(rr_value: float, exp_days: float) -> float:
    if exp_days is None or not np.isfinite(exp_days) or exp_days <= 0:
        return float("nan")
    return float(rr_value / exp_days)


def adj_ev(struct_ev: float, market_score: int, macro_caution: bool, risk_on: bool) -> float:
    x = float(struct_ev)

    if market_score >= 75:
        x *= 1.10
    elif market_score >= 55:
        x *= 1.00
    elif market_score >= 45:
        x *= 0.95
    else:
        x *= 0.90

    if macro_caution and not risk_on:
        x *= 0.90
    elif macro_caution and risk_on:
        x *= 0.95

    return float(clamp(x, -2.0, 3.0))
