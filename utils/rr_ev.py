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

    # 正統EV（因子圧縮）: RR × TrendStrength × PullbackQuality
    # ※内部スケールは後段で正規化（表示・閾値と整合）
    structural_ev_raw = rr * float(setup.trend_strength) * float(setup.pullback_quality)

    # --- AdjEV（内部スケール正規化） ---
    # 目的：AdjEV を現実的レンジに収め、閾値0.50の意味を安定させる
    # 目安：良い=0.6〜1.0 / 極端値は上限1.2に飽和
    adj = float(structural_ev_raw) * 0.35

    # 後段補正（軽く／同一スケール）
    if setup.gu:
        adj -= 0.10
    if macro_on:
        adj -= 0.08

    # 内部上限・下限（運用安定のため）
    adj = float(clamp(adj, -0.50, 1.20))

    return EVInfo(
        rr=rr,
        structural_ev=float(structural_ev_raw),
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
