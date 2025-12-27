from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.features import Features
from utils.setup import SetupResult
from utils.entry import EntryPlan

@dataclass(frozen=True)
class TradePlan:
    stop: float
    tp1: float
    tp2: float
    rr: float
    expected_days: float
    r_per_day: float
    pwin: float
    ev: float
    adjev: float
    regime_mult: float

def _norm(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x) or hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def regime_multiplier(mkt_score: int, delta3d: int, major_event_tomorrow: bool) -> float:
    mult = 1.0
    if mkt_score >= 60 and delta3d >= 0:
        mult *= 1.05
    if delta3d <= -5:
        mult *= 0.70
    if major_event_tomorrow:
        mult *= 0.75
    return float(np.clip(mult, 0.4, 1.15))

def compute_trade_plan(
    hist: pd.DataFrame,
    f: Features,
    s: SetupResult,
    e: EntryPlan,
    mkt_score: int,
    delta3d: int,
    sector_rank: int,
    major_event_tomorrow: bool,
) -> Optional[TradePlan]:
    atr = float(f.atr)
    if not (np.isfinite(atr) and atr > 0):
        return None

    entry = float(e.in_center)
    in_low = float(e.in_low)

    if s.setup == "A":
        stop = in_low - 0.7 * atr
        swing_low = float(hist["Low"].astype(float).tail(12).min())
        stop = min(stop, swing_low - 0.2 * atr)
    else:
        stop = float(s.breakout_line) - 1.0 * atr
        swing_low = float(hist["Low"].astype(float).tail(12).min())
        stop = min(stop, swing_low - 0.2 * atr)

    risk = entry - stop
    if not (np.isfinite(risk) and risk > 0):
        return None

    slope = float(f.slope20_5d) if np.isfinite(f.slope20_5d) else 0.0
    slope_n = _norm(slope, 0.0, 0.05)
    rs_n = _norm(f.rs20, -0.05, 0.10)
    sec_n = _norm(6 - sector_rank, 0, 5)

    mom = 1.0 + 0.35 * slope_n + 0.25 * rs_n + 0.15 * sec_n
    reg = 0.90 + 0.20 * _norm(mkt_score, 45, 75) - 0.10 * _norm(-delta3d, 0, 10)
    reg = float(np.clip(reg, 0.80, 1.10))

    atr_mult = float(np.clip(2.0 * mom * reg, 1.8, 4.2))
    tp2 = entry + atr_mult * atr
    tp1 = entry + 1.5 * risk

    tp2 = max(tp2, entry + 3.0 * risk)
    tp2 = min(tp2, entry + 6.0 * risk)

    rr = (tp2 - entry) / risk

    k = 1.0 + 0.6 * slope_n
    expected_days = (tp2 - entry) / (k * atr)
    expected_days = float(np.clip(expected_days, 1.0, 10.0))
    r_per_day = float(rr / expected_days) if expected_days > 0 else 0.0

    trend_part = _norm(slope, 0.0, 0.05) * 0.35 + _norm((f.price / f.ma20) - 1.0, -0.02, 0.03) * 0.15
    rsi_part = _norm(f.rsi, 40, 62) * 0.15
    rs_part = _norm(f.rs20, -0.05, 0.10) * 0.20
    sec_part = _norm(6 - sector_rank, 0, 5) * 0.10
    liq_part = _norm(f.turnover_ma20, 2e8, 2e9) * 0.10
    gap_penalty = 0.40 if e.gu_flag else (0.25 if e.in_distance_atr > 0.8 else 0.0)

    pwin = float(np.clip(0.25 + trend_part + rsi_part + rs_part + sec_part + liq_part - gap_penalty, 0.0, 1.0))

    ev = pwin * rr - (1.0 - pwin) * 1.0
    mult = regime_multiplier(mkt_score, delta3d, major_event_tomorrow)
    adjev = ev * mult

    return TradePlan(
        stop=float(stop),
        tp1=float(tp1),
        tp2=float(tp2),
        rr=float(rr),
        expected_days=float(expected_days),
        r_per_day=float(r_per_day),
        pwin=float(pwin),
        ev=float(ev),
        adjev=float(adjev),
        regime_mult=float(mult),
    )
