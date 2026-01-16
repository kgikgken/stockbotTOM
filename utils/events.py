from __future__ import annotations

import os
from typing import List, Dict, Optional, Tuple

import pandas as pd

from utils.util import parse_event_datetime_jst

EVENTS_PATH = "events.csv"

# Important macro keywords to treat as warning day
IMPORTANT_KEYWORDS = (
    "FOMC",
    "CPI",
    "雇用統計",
    "日銀",
    "BOJ",
    "GDP",
    "PPI",
    "ISM",
    "利上げ",
    "金利",
)


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    events: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        date_str = str(row.get("date", "")).strip()
        time_str = str(row.get("time", "")).strip()
        dt_str = str(row.get("datetime", "")).strip()

        if not label:
            continue
        events.append({"label": label, "kind": kind, "date": date_str, "time": time_str, "datetime": dt_str})
    return events


def upcoming_events(today_date, window_days: int = 2) -> List[Tuple[str, str, int]]:
    """Return list of (label, dt_disp, delta_days) for events in [-1, window_days]."""
    out: List[Tuple[str, str, int]] = []
    for ev in load_events():
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if -1 <= delta <= window_days:
            out.append((str(ev.get("label", "")), dt.strftime("%Y-%m-%d %H:%M JST"), int(delta)))
    # sort by date
    out.sort(key=lambda x: x[1])
    return out


def is_macro_caution_day(today_date) -> Tuple[bool, List[str]]:
    """Macro caution ON if any upcoming event matches important keywords."""
    evs = upcoming_events(today_date, window_days=2)
    if not evs:
        return False, []

    hits: List[str] = []
    for label, dt_disp, _delta in evs:
        key = label
        if any(k.lower() in label.lower() for k in IMPORTANT_KEYWORDS):
            hits.append(f"{key}（{dt_disp}）")
        else:
            # also treat explicit kind=macro if provided
            # (kept flexible without hard dependency)
            hits.append(f"{key}（{dt_disp}）")

    # If events exist but none are macro, we still return caution False.
    macro = any(any(k.lower() in label.lower() for k in IMPORTANT_KEYWORDS) for label, _dt, _d in evs)
    return macro, hits
