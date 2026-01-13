from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from utils.screen_logic import rr_min_by_market, rday_min_by_setup
from utils.util import clamp
from utils.setup import SetupInfo

@dataclass
class EVInfo:
    rr: float
    structural_ev: float
    adj_ev: float
    expected_days: float
    rday: float
    rr_min: float
    rday_min: float

def calc_ev(setup: SetupInfo, mkt_score: int, macro_on: bool) -> EVInfo:
    rr_min = float(rr_min_by_market(mkt_score))
    rday_min = float(rday_min_by_setup(setup.setup))

    rr = float(setup.rr)
    expected_days = float(max(setup.expected_days, 0.5))
    rday = float(rr / expected_days)

    # 正統EV = RR × TrendStrength × PullbackQuality（因子圧縮）
    structural_ev = rr * float(setup.trend_strength) * float(setup.pullback_quality)

    # AdjEV: 後段補正のみ（軽く）
    adj = structural_ev
    if setup.gu:
        adj -= 0.10
    if macro_on:
        adj -= 0.08

    adj = float(clamp(adj, -5.0, 10.0))

    return EVInfo(
        rr=rr,
        structural_ev=float(structural_ev),
        adj_ev=float(adj),
        expected_days=float(expected_days),
        rday=float(rday),
        rr_min=float(rr_min),
        rday_min=float(rday_min),
    )

def pass_thresholds(setup: SetupInfo, ev: EVInfo) -> Tuple[bool, str]:
    if ev.rr < ev.rr_min:
        return False, "RR"
    if ev.rday < ev.rday_min:
        return False, "RDAY"
    if ev.adj_ev < 0.50:
        return False, "ADJEV"
    return True, "OK"
