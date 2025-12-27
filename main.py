from __future__ import annotations

import os

from utils.util import jst_today_str, jst_today_date
from utils.market import enhance_market_score
from utils.events import build_event_warnings
from utils.position import load_positions, analyze_positions
from utils.screener import run_swing
from utils.report import build_report
from utils.line import send_line

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"

def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)
    if total_asset <= 0:
        total_asset = 2_000_000.0

    swing_result = run_swing(
        universe_path=UNIVERSE_PATH,
        events_path=EVENTS_PATH,
        today_date=today_date,
        mkt_score=int(mkt.get("score", 50)),
        delta3d=int(mkt.get("delta3d", 0)),
    )

    events = build_event_warnings(EVENTS_PATH, today_date)

    report = build_report(
        today_str=today_str,
        mkt=mkt,
        swing_result=swing_result,
        events=events,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    print(report)
    send_line(report, worker_url=os.getenv("WORKER_URL"))

if __name__ == "__main__":
    main()
