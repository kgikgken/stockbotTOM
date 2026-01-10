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
from utils.state import load_state, bump_weekly_new, save_state


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
    pos_text, total_asset_est, _weekly_from_positions = analyze_positions_summary(
        pos_df, today_date=today_date, mkt_score=int(mkt["score"])
    )
    if not (np.isfinite(total_asset_est) and total_asset_est > 0):
        total_asset_est = 2_000_000.0

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
        weekly_new_count=int(state.get("weekly_new", 0)),
        total_asset=float(total_asset_est),
        positions_text=pos_text,
        screening=screening,
    )

    print(report)
    send_line_text(report, worker_url=WORKER_URL)

    # Machine-mode counter: count 1 when new entries are allowed and at least one non-GU candidate exists
    try:
        if not screening.get("no_trade", False):
            cands = screening.get("candidates", []) or []
            any_actionable = any((not bool(c.get("gu", False))) for c in cands)
            if any_actionable:
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
