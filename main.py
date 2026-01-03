from __future__ import annotations

import os

from utils.util import jst_today_str, jst_today_date
from utils.market import build_market_context
from utils.position import load_positions, analyze_positions
from utils.report import build_report_text
from utils.line import send_line_text


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = build_market_context(today_date=today_date)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt["score"]))

    report = build_report_text(
        today_str=today_str,
        today_date=today_date,
        market=mkt,
        positions_text=pos_text,
        total_asset=total_asset,
        universe_path=UNIVERSE_PATH,
        events_path=EVENTS_PATH,
    )

    # GitHub Actions log
    print(report)

    # LINE
    worker_url = os.getenv("WORKER_URL", "")
    send_line_text(worker_url=worker_url, text=report)


if __name__ == "__main__":
    main()