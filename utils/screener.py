from __future__ import annotations

from typing import Dict, List
import os
import pandas as pd
import numpy as np

from utils.market import rr_min_for_market, leverage_for_market
from utils.diversify import apply_diversification
from utils.screen_logic import build_raw_candidates
from utils.state import load_state

MAX_LINE = 5
MAX_PER_WEEK = 3
GU_RATIO_MAX = 0.40

def _load_universe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def run_screening(
    universe_path: str,
    today_date,
    mkt: Dict[str, object],
    macro_on: bool,
    events: List[Dict[str, str]] | None,
    state: Dict[str, object] | None,
) -> Dict[str, object]:
    mkt_score = int(mkt.get("score", 50) or 50)
    delta3 = float(mkt.get("delta3", 0.0) or 0.0)
    rr_min = rr_min_for_market(mkt_score)

    # NO-TRADE market conditions
    no_trade_reasons: List[str] = []
    if mkt_score < 45:
        no_trade_reasons.append("MarketScore<45")
    if delta3 <= -5 and mkt_score < 55:
        no_trade_reasons.append("Δ3d<=-5 & score<55")

    # weekly cap
    state = state or {}
    weekly_new = int(state.get("weekly_new", 0) or 0)
    if weekly_new >= MAX_PER_WEEK:
        no_trade_reasons.append("weekly_new_limit>=3")

    # if hard no-trade, return early (single return path is still fine here)
    if no_trade_reasons:
        return {
            "no_trade": True,
            "no_trade_reasons": no_trade_reasons,
            "candidates": [],
            "stats": {"raw_n": 0, "final_n": 0, "avg_adjev": 0.0, "gu_ratio": 0.0},
        }

    uni = _load_universe(universe_path)
    if uni is None or len(uni) == 0:
        return {
            "no_trade": True,
            "no_trade_reasons": ["universe_missing_or_empty"],
            "candidates": [],
            "stats": {"raw_n": 0, "final_n": 0, "avg_adjev": 0.0, "gu_ratio": 0.0},
        }

    raw, st = build_raw_candidates(universe=uni, today_date=today_date, mkt_score=mkt_score, macro_on=macro_on)

    # Sort by AdjEV * R/day (primary), then AdjEV, then RR
    raw.sort(key=lambda x: (x.get("adjev", -999.0) * x.get("rday", 0.0), x.get("adjev", -999.0), x.get("rr", 0.0)), reverse=True)

    diversified = apply_diversification(raw, max_per_sector=2, corr_max=0.75)

    # macro candidate cap
    cap = 2 if macro_on else MAX_LINE
    candidates = diversified[:cap]

    # GU ratio check (NO-TRADE if too high)
    gu_ratio = float(np.mean([1.0 if c.get("gu", False) else 0.0 for c in candidates])) if candidates else 0.0
    if gu_ratio > GU_RATIO_MAX:
        return {
            "no_trade": True,
            "no_trade_reasons": ["GU_ratio_high"],
            "candidates": [],
            "stats": {"raw_n": int(st.get("raw_n", 0)), "final_n": 0, "avg_adjev": 0.0, "gu_ratio": gu_ratio},
        }

    avg_adjev = float(np.mean([float(c.get("adjev", 0.0)) for c in candidates])) if candidates else 0.0
    if avg_adjev < 0.50:
        return {
            "no_trade": True,
            "no_trade_reasons": ["avgAdjEV<0.5"],
            "candidates": [],
            "stats": {"raw_n": int(st.get("raw_n", 0)), "final_n": 0, "avg_adjev": avg_adjev, "gu_ratio": gu_ratio},
        }

    # fill leverage in market context
    mkt["lev"] = leverage_for_market(mkt_score, macro_on)

    return {
        "no_trade": False,
        "no_trade_reasons": [],
        "candidates": candidates[:MAX_LINE],
        "stats": {
            "raw_n": int(st.get("raw_n", 0)),
            "final_n": int(len(candidates[:MAX_LINE])),
            "avg_adjev": avg_adjev,
            "gu_ratio": gu_ratio,
            "rr_min": rr_min,
        },
    }
