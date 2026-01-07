# ============================================
# main.py
# stockbotTOM (Swing 1-7d) - Clean Version
# - NO-TRADE fully mechanical
# - AdjEV filter, RR min by market, event cap(=2), weekly cap
# - Setup A split to A1/A2
# - Natural RR / R-day distribution (not fixed)
# - LINE delivery via Cloudflare Worker (WORKER_URL)
# ============================================

from __future__ import annotations

import os
import sys
import json
import traceback
from dataclasses import asdict
from typing import List, Optional, Dict, Any

import pandas as pd

from utils.util import jst_today_str, jst_now_str
from utils.market import compute_market_context
from utils.sector import compute_top_sectors_5d
from utils.events import load_events, detect_macro_risk
from utils.position import load_positions, analyze_positions
from utils.screener import run_screener
from utils.report import build_daily_report_text
from utils.line import send_line_message


UNIVERSE_PATH = os.getenv("UNIVERSE_PATH", "universe_jpx.csv")
POSITIONS_PATH = os.getenv("POSITIONS_PATH", "positions.csv")
EVENTS_PATH = os.getenv("EVENTS_PATH", "events.csv")
WEEKLY_STATE_PATH = os.getenv("WEEKLY_STATE_PATH", "weekly_state.json")

WORKER_URL = os.getenv("WORKER_URL", "").strip()


def _load_universe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Universe not found: {path}")
    df = pd.read_csv(path)
    # expected columns (flexible):
    # ticker, name, sector, earnings_date (optional)
    if "ticker" not in df.columns:
        raise ValueError("universe_jpx.csv must include 'ticker' column")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


def _load_weekly_state(path: str) -> Dict[str, Any]:
    # Track weekly new entries count: resets on Monday (JST) or when week key changes.
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_weekly_state(path: str, state: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        # Do not crash screening if state save fails
        pass


def _current_week_key_jst() -> str:
    # ISO week key based on JST date
    # Example: 2026-W01
    import datetime
    from utils.util import JST

    d = datetime.datetime.now(JST).date()
    y, w, _ = d.isocalendar()
    return f"{y}-W{int(w):02d}"


def _get_weekly_new_count(state: Dict[str, Any], week_key: str) -> int:
    if state.get("week_key") != week_key:
        return 0
    return int(state.get("weekly_new_count", 0) or 0)


def _set_weekly_new_count(state: Dict[str, Any], week_key: str, count: int) -> None:
    state["week_key"] = week_key
    state["weekly_new_count"] = int(count)


def main() -> int:
    today = jst_today_str()
    now_str = jst_now_str()

    # --- Load core inputs
    universe = _load_universe(UNIVERSE_PATH)
    positions = load_positions(POSITIONS_PATH)
    events = load_events(EVENTS_PATH)

    # --- Weekly state
    week_key = _current_week_key_jst()
    weekly_state = _load_weekly_state(WEEKLY_STATE_PATH)
    weekly_new_count = _get_weekly_new_count(weekly_state, week_key)

    # --- Market / Sector / Macro
    market_ctx = compute_market_context()
    top_sectors = compute_top_sectors_5d(universe=universe, lookback_days=5)

    macro_risk = detect_macro_risk(
        events=events,
        within_days=2,
        now_jst=now_str,
    )

    # --- Analyze positions (risk / lot accident)
    pos_summary = analyze_positions(positions=positions)

    # --- Run screener
    screen_out = run_screener(
        universe=universe,
        positions=positions,
        events=events,
        market_ctx=market_ctx,
        top_sectors=top_sectors,
        macro_risk=macro_risk,
        weekly_new_count=weekly_new_count,
    )

    # Update weekly state if "new entries allowed and executed" is conceptually increased.
    # In this bot, we only recommend; user executes. So we increment only when user opts in.
    # BUT to keep it fully mechanical, we treat "candidates produced under NEW-OK" as "1 slot used"
    # only if we were not NO-TRADE and there is at least one EXEC_NOW or LIMIT_WAIT.
    # You can change to manual later.
    if screen_out.weekly_new_count_next is not None:
        _set_weekly_new_count(weekly_state, week_key, screen_out.weekly_new_count_next)
        _save_weekly_state(WEEKLY_STATE_PATH, weekly_state)

    # --- Build report text
    report_text = build_daily_report_text(
        date_str=today,
        market_ctx=market_ctx,
        top_sectors=top_sectors,
        events=events,
        macro_risk=macro_risk,
        weekly_new_count=_get_weekly_new_count(weekly_state, week_key),
        screen_out=screen_out,
        pos_summary=pos_summary,
    )

    # --- Send LINE via Worker
    if not WORKER_URL:
        print(report_text)
        print("\n[WARN] WORKER_URL is empty. Printed report instead of sending.")
        return 0

    ok = send_line_message(worker_url=WORKER_URL, message=report_text)
    if not ok:
        print(report_text)
        print("\n[ERROR] LINE send failed. Printed report for debugging.")
        return 2

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("[FATAL] main.py crashed:", str(e))
        traceback.print_exc()
        # Try to send crash notice if possible
        try:
            if WORKER_URL:
                msg = "[FATAL] stockbotTOM crashed\n" + jst_now_str() + "\n" + str(e)
                send_line_message(worker_url=WORKER_URL, message=msg)
        except Exception:
            pass
        sys.exit(1)