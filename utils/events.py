from __future__ import annotations

import os
from typing import List, Dict, Tuple
import pandas as pd

from utils.util import parse_event_datetime_jst, jst_today_date


EVENTS_PATH = "events.csv"


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
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
        kind = str(row.get("kind", "")).strip()
        date_str = str(row.get("date", "")).strip()
        time_str = str(row.get("time", "")).strip()
        dt_str = str(row.get("datetime", "")).strip()
        out.append({"label": label, "kind": kind, "date": date_str, "time": time_str, "datetime": dt_str})
    return out


def build_event_warnings(today_date=None) -> Tuple[List[str], bool]:
    """
    returns: (lines, is_risk_day)
      - is_risk_day: FOMC/日銀 など大イベント近接を True
    """
    if today_date is None:
        today_date = jst_today_date()

    events = load_events()
    lines: List[str] = []
    risk = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days

        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            lines.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

            key = (ev.get("label", "") + " " + ev.get("kind", "")).lower()
            if ("fomc" in key) or ("fed" in key) or ("日銀" in key) or ("boj" in key) or ("cpi" in key):
                risk = True

    if not lines:
        lines.append("- 特になし")
    return lines, risk