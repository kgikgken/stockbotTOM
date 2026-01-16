from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

STATE_PATH = Path("state.json")


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {
            "week_id": "",
            "weekly_new": 0,
            "distortion_off_until": "",
            "distortion_recent_r": [],
            "distortion_recent_gap": [],
        }
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "week_id": "",
            "weekly_new": 0,
            "distortion_off_until": "",
            "distortion_recent_r": [],
            "distortion_recent_gap": [],
        }


def save_state(state: Dict[str, Any]) -> None:
    try:
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _iso_week_id(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def update_weekly_counter(state: Dict[str, Any], today: date) -> Dict[str, Any]:
    week_id = _iso_week_id(today)
    if state.get("week_id") != week_id:
        state["week_id"] = week_id
        state["weekly_new"] = 0
    save_state(state)
    return state


def inc_weekly_new(state: Dict[str, Any], n: int = 1) -> None:
    state["weekly_new"] = int(state.get("weekly_new", 0)) + int(n)
    save_state(state)
