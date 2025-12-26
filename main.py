from __future__ import annotations

import os
from typing import List

import numpy as np

from utils.util import jst_today_str, jst_today_date
from utils.market import get_market_context
from utils.events import get_event_context
from utils.sector import get_sector_rank_map
from utils.screener import run_swing_screen
from utils.position import load_positions, analyze_positions
from utils.report import build_report
from utils.line import send_line

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

def calc_max_position(total_asset: float, lev: float) -> float:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0.0
    return float(total_asset * lev)

def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = get_market_context()
    ev_ctx = get_event_context(today_date)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    max_pos_yen = calc_max_position(total_asset, mkt.lev)

    # sector tops (for display)
    sector_tops, _, _ = get_sector_rank_map(top_n=5)

    result = run_swing_screen(today_date=today_date, mkt=mkt, ev_ctx=ev_ctx)

    report = build_report(
        today_str=today_str,
        mkt=mkt,
        ev_ctx=ev_ctx,
        sector_tops=sector_tops,
        result=result,
        pos_text=pos_text,
        max_position_yen=max_pos_yen,
    )

    print(report)
    send_line(report, WORKER_URL)

if __name__ == "__main__":
    main()
