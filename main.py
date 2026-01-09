
from __future__ import annotations

import os
import sys
import numpy as np

from utils.util import jst_today_str, jst_today_date
from utils.market import calc_market_context
from utils.events import load_events, build_event_warnings, is_macro_caution
from utils.position import load_positions, analyze_positions_summary
from utils.screener import run_screening
from utils.report import build_report
from utils.line import send_line_text
from utils.state import load_state, update_state_after_run

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    state = load_state(today_date)

    mkt = calc_market_context(today_date=today_date)

    events = load_events(EVENTS_PATH)
    macro_on = is_macro_caution(today_date=today_date, events=events)
    event_warnings = build_event_warnings(today_date=today_date, events=events)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset_est, weekly_new_count = analyze_positions_summary(
        pos_df, today_date=today_date, mkt_score=int(mkt["score"])
    )

    screening = run_screening(
        universe_path=UNIVERSE_PATH,
        today_date=today_date,
        mkt=mkt,
        macro_on=macro_on,
        events=events,
        state=state,
    )

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        macro_on=macro_on,
        event_warnings=event_warnings,
        weekly_new_count=state["weekly_new"],
        total_asset=float(total_asset_est),
        positions_text=pos_text,
        screening=screening,
    )

    print(report)
    send_line_text(report, worker_url=WORKER_URL)

    update_state_after_run(today_date, screening, state)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e, file=sys.stderr)
        raise
