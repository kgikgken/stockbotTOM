from __future__ import annotations

import json
from pathlib import Path
from datetime import date

STATE_PATH = Path("state.json")

def load_state(today_date: date) -> dict:
    y = int(today_date.isocalendar().year)
    w = int(today_date.isocalendar().week)

    s = {}
    if STATE_PATH.exists():
        try:
            s = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            s = {}

    if int(s.get("year", -1)) != y or int(s.get("week", -1)) != w:
        s = {"year": y, "week": w, "weekly_new": 0}
    if "weekly_new" not in s:
        s["weekly_new"] = 0
    return s

def bump_weekly_new(state: dict, n: int = 1) -> None:
    try:
        state["weekly_new"] = int(state.get("weekly_new", 0)) + int(n)
    except Exception:
        state["weekly_new"] = int(n)

def save_state(state: dict) -> None:
    try:
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
