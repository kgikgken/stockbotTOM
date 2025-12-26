from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from utils.util import parse_event_datetime_jst

EVENTS_PATH = "events.csv"

@dataclass
class EventContext:
    warnings: List[str]
    is_major_event_near: bool
    event_multiplier: float

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
        if not label:
            continue
        kind = str(row.get("kind", "")).strip()
        date_str = str(row.get("date", "")).strip()
        time_str = str(row.get("time", "")).strip()
        dt_str = str(row.get("datetime", "")).strip()
        importance = str(row.get("importance", "")).strip()  # 任意列
        events.append({"label": label, "kind": kind, "date": date_str, "time": time_str, "datetime": dt_str, "importance": importance})
    return events

def get_event_context(today_date) -> EventContext:
    events = load_events()
    warns: List[str] = []
    major_near = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue

        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"

            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

            # 「重要」扱い（kind/importance のどれかがそれっぽいなら近いとする）
            k = (ev.get("kind","") or "").lower()
            imp = (ev.get("importance","") or "").lower()
            if any(x in k for x in ["fomc","cpi","pce","boj","gdp","employment","payroll","is m","ism"]) or any(x in imp for x in ["high","major","重要"]):
                if delta in (0,1):  # 前日/当日あたりを重視
                    major_near = True

    if not warns:
        warns.append("- 特になし")

    # v2.0: 重要イベント前日は 0.75
    mult = 0.75 if major_near else 1.00
    return EventContext(warnings=warns, is_major_event_near=major_near, event_multiplier=mult)
