from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Any

import pandas as pd

from utils.market import MarketSnapshot
from utils.state import inc_weekly_new
from utils.screen_logic import Candidate, build_raw_candidates
from utils.diversify import diversify


@dataclass
class ScreeningResult:
    allow_new: bool
    no_trade_reason: str
    rr_min: float
    adj_ev_min: float
    candidates: List[Candidate]


def _rr_min_by_market(score: float) -> float:
    # v2.3: 強いほどRR下限を緩める（ただし下限1.8）
    if score >= 70:
        return 1.8
    if score >= 55:
        return 2.0
    return 2.2


def run_screening(today_date: date, universe_path: str, market: MarketSnapshot, state: Dict[str, Any], total_asset: float) -> ScreeningResult:
    # thresholds
    rr_min = _rr_min_by_market(market.market_score)
    adj_ev_min = 0.50
    rday_min_by_setup = {
        "A1-Strong": 0.50,
        "A1": 0.45,
        "A2": 0.50,
        "B": 0.65,
    }

    # load universe
    try:
        uni = pd.read_csv(universe_path)
    except Exception:
        uni = pd.DataFrame(columns=["ticker", "name", "sector"])

    raw = build_raw_candidates(
        today_date=today_date,
        universe=uni,
        rr_min=rr_min,
        adj_ev_min=adj_ev_min,
        rday_min_by_setup=rday_min_by_setup,
    )

    # NO-TRADE gate
    no_trade = False
    reason = ""
    if market.market_score < 45:
        no_trade = True
        reason = "地合いが弱い（MarketScore<45）"
    if market.delta_score_3d <= -5 and market.market_score < 55:
        no_trade = True
        reason = "地合い悪化（ΔMarketScore_3d≤-5 かつ MarketScore<55）"

    # avgAdjEV gate (after raw)
    avg_adj = sum(c.adj_ev for c in raw[:20]) / max(1, min(20, len(raw))) if raw else 0.0
    if avg_adj < 0.5:
        no_trade = True
        reason = "期待効率の平均が低い（平均<0.5）"

    # diversify & cap
    final = diversify(raw, max_per_sector=2, corr_limit=0.75, max_total=5)

    # "非常口"：NO-TRADEでもTier0を最大1銘柄だけ残す
    if no_trade:
        tier0 = [c for c in final if c.setup == "A1-Strong"]
        if tier0:
            final = tier0[:1]
            no_trade = False
            reason = "（非常口）Tier0のみ表示"
        else:
            final = []

    allow_new = not no_trade

    # weekly counter (only if actual candidates)
    if allow_new and final:
        inc_weekly_new(state, 0)  # 現状は表示のみ。実エントリーが確定したら加算。

    return ScreeningResult(
        allow_new=allow_new,
        no_trade_reason=reason,
        rr_min=rr_min,
        adj_ev_min=adj_ev_min,
        candidates=final,
    )
