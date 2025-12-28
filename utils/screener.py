from __future__ import annotations

import pandas as pd
from typing import Dict, List

from utils.market import evaluate_market
from utils.events import load_events
from utils.setup import judge_setup
from utils.entry import judge_entry_action
from utils.rr_ev import calc_rr_ev_speed
from utils.diversify import apply_diversification


UNIVERSE_PATH = "universe_jpx.csv"


def run_screening() -> Dict:
    """
    スクリーニング全体統括
    """
    universe = pd.read_csv(UNIVERSE_PATH)

    # --- 地合い判定 ---
    market = evaluate_market()

    # --- NO-TRADE 判定 ---
    no_trade_reason = None
    if market["no_trade"]:
        no_trade_reason = market["no_trade_reason"]

    # --- イベント ---
    events = load_events()

    candidates: List[Dict] = []
    watchlist: List[Dict] = []

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        sector = row.get("sector", "不明")

        setup = judge_setup(ticker)
        if not setup["valid"]:
            continue

        entry = judge_entry_action(ticker, setup)
        rr_ev = calc_rr_ev_speed(ticker, setup, entry, market)

        if not rr_ev["valid"]:
            watchlist.append({**rr_ev, "ticker": ticker, "sector": sector})
            continue

        candidates.append({
            "ticker": ticker,
            "sector": sector,
            **setup,
            **entry,
            **rr_ev,
        })

    # --- 分散制御 ---
    final, dropped = apply_diversification(candidates)

    watchlist.extend(dropped)

    return {
        "market": market,
        "events": events,
        "no_trade": no_trade_reason,
        "candidates": final,
        "watchlist": watchlist,
    }