from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from utils.util import JST

def parse_event_datetime_jst(dt_str: str | None, date_str: str | None, time_str: str | None) -> Optional[datetime]:
    dt_str = (dt_str or "").strip()
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()

    if dt_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_str, fmt).replace(tzinfo=JST)
            except Exception:
                pass

    if date_str and time_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(f"{date_str} {time_str}", fmt).replace(tzinfo=JST)
            except Exception:
                pass

    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=JST)
        except Exception:
            return None

    return None

def load_events(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        out.append(
            {
                "label": label,
                "kind": str(row.get("kind", "")).strip(),
                "date": str(row.get("date", "")).strip(),
                "time": str(row.get("time", "")).strip(),
                "datetime": str(row.get("datetime", "")).strip(),
            }
        )
    return out

def build_event_warnings(path: str, today_date) -> List[str]:
    warns: List[str] = []
    for ev in load_events(path):
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if -1 <= delta <= 2:
            when = "直近" if delta < 0 else ("本日" if delta == 0 else f"{delta}日後")
            warns.append(f"⚠ {ev['label']}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")
    return warns if warns else ["- 特になし"]

def is_major_event_tomorrow(path: str, today_date) -> bool:
    for ev in load_events(path):
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        if (dt.date() - today_date).days == 1:
            return True
    return False
