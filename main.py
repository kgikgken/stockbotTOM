from __future__ import annotations

import os

from utils.util import jst_today_str, jst_today_date
from utils.market import build_market_snapshot
from utils.state import load_state, update_weekly_counter
from utils.position import load_positions, analyze_positions
from utils.screener import run_screening
from utils.report import build_line_report
from utils.line import send_line


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    st = load_state()
    st = update_weekly_counter(st, today_date)

    mkt = build_market_snapshot(today_date=today_date, events_path=EVENTS_PATH)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt.market_score))

    screening = run_screening(
        today_date=today_date,
        universe_path=UNIVERSE_PATH,
        market=mkt,
        state=st,
        total_asset=total_asset,
    )

    report = build_line_report(today_str=today_str, market=mkt, screening=screening, positions_text=pos_text)

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
