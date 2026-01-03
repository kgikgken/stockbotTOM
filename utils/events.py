from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

from utils.util import parse_event_datetime_jst


IMPORTANT_KEYWORDS = (
    "FOMC",
    "日銀",
    "金融政策",
    "雇用統計",
    "CPI",
    "GDP",
    "PCE",
)


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
        kind = str(row.get("kind", "")).strip()
        date_str = str(row.get("date", "")).strip()
        time_str = str(row.get("time", "")).strip()
        dt_str = str(row.get("datetime", "")).strip()
        if not label:
            continue
        out.append({"label": label, "kind": kind, "date": date_str, "time": time_str, "datetime": dt_str})
    return out


def is_important_event(label: str) -> bool:
    s = (label or "").strip()
    if not s:
        return False
    return any(k in s for k in IMPORTANT_KEYWORDS)


def build_event_warnings(today_date, events_path: str) -> Tuple[List[str], bool]:
    """
    Returns: (lines, event_near)
      - event_near: 重要イベントが「当日〜2日後」にある
    """
    events = load_events(events_path)
    warns: List[str] = []
    event_near = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days

        if -1 <= delta <= 2:
            when = "直近"
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"

            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            line = f"⚠ {ev['label']}（{dt_disp} / {when}）"
            warns.append(line)

            if is_important_event(ev.get("label", "")) and 0 <= delta <= 2:
                event_near = True

    if not warns:
        warns.append("- 特になし")

    return warns, event_near


def event_risk_multiplier(event_near: bool) -> float:
    # 仕様：イベント接近は強制減衰
    return 0.75 if event_near else 1.0