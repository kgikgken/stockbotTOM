from __future__ import annotations

import os
import sys
import numpy as np

from utils.util import jst_today_str, jst_today_date
from utils.events import load_events, build_event_warnings, is_macro_caution
from utils.market import calc_market_context
from utils.position import load_positions, analyze_positions
from utils.screener import run_screening
from utils.report import build_report
from utils.line import send_line_text
from utils.state import load_state, save_state, bump_weekly_new


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    state = load_state(today_date)

    # Market / macro
    mkt = calc_market_context(today_date=today_date)

    events = load_events(EVENTS_PATH)
    macro_on = is_macro_caution(today_date=today_date, events=events)
    event_warnings = build_event_warnings(today_date=today_date, events=events)

    # Positions (optional)
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, today_date=today_date, mkt=mkt, macro_on=macro_on)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    # Screening
    result = run_screening(
        universe_path=UNIVERSE_PATH,
        today_date=today_date,
        mkt=mkt,
        macro_on=macro_on,
        events=events,
        state=state,
    )

    text = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        macro_on=macro_on,
        event_warnings=event_warnings,
        weekly_new=int(state.get("weekly_new", 0)),
        total_asset=float(total_asset),
        positions_text=pos_text,
        screening=result,
    )

    print(text)
    send_line_text(text, worker_url=WORKER_URL)

    # weekly counter: bump only if today is not NO-TRADE and there is at least one non-GU actionable candidate
    try:
        if not bool(result.get("no_trade", False)):
            cands = result.get("candidates", []) or []
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
