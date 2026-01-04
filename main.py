from __future__ import annotations

import os
from typing import List, Dict

import numpy as np
import requests

from utils.util import jst_today_str, jst_today_date
from utils.line import send_line
from utils.events import load_events, build_event_section, detect_macro_caution
from utils.market import build_market_context
from utils.sector import build_sector_ranking
from utils.position import load_positions, analyze_positions, calc_weekly_new_count
from utils.screener import run_swing_screening
from utils.report import build_daily_report


# =========================
# Paths / Env
# =========================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # --- Market ---
    mkt = build_market_context()

    # --- Events ---
    events = load_events(EVENTS_PATH)
    event_lines = build_event_section(events, today_date)
    macro_caution = detect_macro_caution(events, today_date)  # イベント接近なら ON

    # --- Positions ---
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt["score"])
    weekly_new = calc_weekly_new_count(pos_df, today_date=today_date)  # entry_date が無ければ 0

    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    # --- Sector (説明用) ---
    sector_rank = build_sector_ranking(universe_path=UNIVERSE_PATH, top_n=5)

    # --- Screening ---
    swing_result = run_swing_screening(
        universe_path=UNIVERSE_PATH,
        today_date=today_date,
        market=mkt,
        macro_caution=macro_caution,
        weekly_new=weekly_new,
    )

    # --- Report ---
    report = build_daily_report(
        today_str=today_str,
        today_date=today_date,
        market=mkt,
        macro_caution=macro_caution,
        weekly_new=weekly_new,
        total_asset=float(total_asset),
        sector_rank=sector_rank,
        event_lines=event_lines,
        swing=swing_result,
        pos_text=pos_text,
    )

    print(report)
    send_line(report, worker_url=WORKER_URL)


if __name__ == "__main__":
    main()