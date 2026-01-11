from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

from utils.util import parse_event_datetime_jst

@dataclass
class Event:
    label: str
    kind: str
    date: str
    time: str
    datetime: str

def load_events(path: str) -> List[Event]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    out: List[Event] = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        out.append(Event(
            label=label,
            kind=str(row.get("kind", "")).strip(),
            date=str(row.get("date", "")).strip(),
            time=str(row.get("time", "")).strip(),
            datetime=str(row.get("datetime", "")).strip(),
        ))
    return out

def macro_warning_block(events: List[Event], today_date) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    macro_on = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.datetime, ev.date, ev.time)
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            macro_on = True
            lines.append(f"・{ev.label}（{dt.strftime('%Y-%m-%d %H:%M JST')}）")

    return macro_on, (lines if lines else [])
