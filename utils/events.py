from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

from utils.util import parse_event_datetime_jst

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

def build_event_section(today_date) -> Tuple[List[str], bool]:
    """
    直近（-1〜+2日）のイベントを警告に出す。
    Macro警戒: kind == 'macro' のイベントが存在する場合 True
    """
    events = load_events()
    warns: List[str] = []
    macro_on = False

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
            if str(ev.get("kind", "")).lower() == "macro":
                macro_on = True

    if not warns:
        warns.append("- 特になし")

    return warns, macro_on
