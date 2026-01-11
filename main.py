from __future__ import annotations

import os
import sys
import numpy as np

from utils.util import jst_today_str, jst_today_date
from utils.state import load_state, save_state, bump_weekly_new
from utils.market import calc_market_context, leverage_for_market
from utils.events import load_events, build_event_items, is_macro_caution
from utils.position import load_positions, analyze_positions
from utils.screener import run_screening
from utils.report import build_report
from utils.line import send_line_text

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"

def main() -> None:
    worker_url = os.getenv("WORKER_URL")

    today_str = jst_today_str()
    today_date = jst_today_date()

    state = load_state(today_date)

    mkt = calc_market_context(today_date=today_date)
    mkt_score = int(mkt.get("score", 50) or 50)

    events = load_events(EVENTS_PATH)
    event_items = build_event_items(today_date=today_date, events=events)
    macro_on = is_macro_caution(today_date=today_date, events=events)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, today_date=today_date, mkt=mkt, macro_on=macro_on)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    lev = leverage_for_market(mkt_score, macro_on=macro_on)
    mkt["lev"] = lev

    screening = run_screening(
        universe_path=UNIVERSE_PATH,
        today_date=today_date,
        mkt=mkt,
        macro_on=macro_on,
        state=state,
    )

    text = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        macro_on=macro_on,
        event_items=event_items,
        weekly_new=int(state.get("weekly_new", 0) or 0),
        positions_text=pos_text,
        screening=screening,
    )

    print(text)
    send_line_text(text, worker_url=worker_url)

    try:
        if not bool(screening.get("no_trade", False)):
            cands = screening.get("candidates", []) or []
            if any((not bool(c.get("gu", False))) for c in cands):
                bump_weekly_new(state, 1)
        save_state(state)
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", repr(e), file=sys.stderr)
        raise
