import os
from typing import Dict, List

import pandas as pd

from utils.util import parse_event_datetime_jst


def load_events(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
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


def build_event_warnings(events_path: str, today_date) -> List[str]:
    events = load_events(events_path)
    warns: List[str] = []

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

    if not warns:
        warns.append("- 特になし")
    return warns


def is_major_event_day(events_path: str, today_date) -> bool:
    """重要イベント前後は RegimeMultiplier で吸収するためのフラグ"""
    events = load_events(events_path)
    for ev in events:
        label = str(ev.get("label", "")).upper()
        if any(k in label for k in ("FOMC", "日銀", "BOJ", "CPI", "雇用", "GDP", "PCE")):
            dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
            if dt is None:
                continue
            delta = (dt.date() - today_date).days
            if -1 <= delta <= 1:
                return True
    return False