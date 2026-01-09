
from __future__ import annotations

import json
from pathlib import Path
from datetime import date

STATE_PATH = Path("state.json")

def load_state(today_date: date) -> dict:
    if STATE_PATH.exists():
        try:
            s = json.loads(STATE_PATH.read_text())
        except Exception:
            s = {}
    else:
        s = {}

    if s.get("year") != today_date.isocalendar().year or s.get("week") != today_date.isocalendar().week:
        s["year"] = today_date.isocalendar().year
        s["week"] = today_date.isocalendar().week
        s["weekly_new"] = 0

    return s

def update_state_after_run(today_date: date, screening: dict, state: dict) -> None:
    if not screening.get("no_trade", False):
        state["weekly_new"] = int(state.get("weekly_new", 0))

    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
