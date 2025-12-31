from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

from utils.util import jst_today_str, jst_today_date
from utils.market import build_market_context
from utils.sector import top_sectors_5d
from utils.events import build_event_warnings
from utils.position import load_positions, analyze_positions
from utils.screener import run_screening
from utils.report import build_report
from utils.line import send_line_message


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"

WORKER_URL = os.getenv("WORKER_URL")


def _load_universe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e

    # Accept "ticker" or "code"
    if "ticker" not in df.columns and "code" not in df.columns:
        raise RuntimeError("universe_jpx.csv must have 'ticker' or 'code' column")

    # Normalize ticker column name
    if "ticker" not in df.columns and "code" in df.columns:
        df = df.rename(columns={"code": "ticker"})

    # Optional cols
    if "name" not in df.columns:
        df["name"] = df["ticker"].astype(str)
    if "sector" not in df.columns and "industry_big" in df.columns:
        df["sector"] = df["industry_big"]
    if "sector" not in df.columns:
        df["sector"] = "不明"

    return df


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    uni = _load_universe(UNIVERSE_PATH)

    mkt = build_market_context()

    sectors = top_sectors_5d(universe_path=UNIVERSE_PATH, top_n=5)
    events = build_event_warnings(events_path=EVENTS_PATH, today_date=today_date)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset_est = analyze_positions(pos_df, mkt_score=int(mkt["score"]))

    # Safety fallback
    if total_asset_est is None or total_asset_est <= 0:
        total_asset_est = 2_000_000.0

    screening = run_screening(
        universe_df=uni,
        today_date=today_date,
        market_ctx=mkt,
        total_asset=total_asset_est,
    )

    report = build_report(
        today_str=today_str,
        market_ctx=mkt,
        sectors=sectors,
        events=events,
        screening=screening,
        pos_text=pos_text,
        total_asset=total_asset_est,
    )

    print(report)

    # Send LINE (same "delivered" spec)
    send_line_message(text=report, worker_url=WORKER_URL)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Never crash silently in Actions; print error block
        print("=== stockbotTOM ERROR ===")
        print(str(e))
        raise