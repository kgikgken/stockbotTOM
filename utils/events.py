from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd
from utils.util import safe_read_csv, parse_date_yyyy_mm_dd

@dataclass
class Event:
    date: object
    name: str

def load_events(path: str) -> List[Event]:
    df = safe_read_csv(path)
    out = []
    for _, r in df.iterrows():
        d = parse_date_yyyy_mm_dd(r.get("date"))
        if d:
            out.append(Event(d, str(r.get("name", ""))))
    return out

def is_event_eve(events: List[Event], today: pd.Timestamp) -> bool:
    return any(e.date == (today + pd.Timedelta(days=1)).date() for e in events)