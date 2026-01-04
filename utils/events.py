from __future__ import annotations

import os
import pandas as pd
from typing import List, Dict

from utils.util import parse_event_datetime_jst


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
        out.append({
            "label": label,
            "kind": str(row.get("kind", "")).strip(),
            "date": str(row.get("date", "")).strip(),
            "time": str(row.get("time", "")).strip(),
            "datetime": str(row.get("datetime", "")).strip(),
        })
    return out


def build_event_warnings(today_date, path: str) -> List[str]:
    evs = load_events(path)
    warns: List[str] = []
    for ev in evs:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            when = "直近" if delta < 0 else ("本日" if delta == 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")
    if not warns:
        warns.append("- 特になし")
    return warns


def is_macro_danger(today_date, path: str) -> bool:
    """
    “イベント接近”をマクロ警戒ONにする（仕様：イベント日は取らないも正解）
    - 2日以内に重要イベントがあると danger
    """
    evs = load_events(path)
    for ev in evs:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if 0 <= delta <= 2:
            # kindが空でも重要扱い（ユーザーが入れてるイベントは警戒したい）
            return True
    return False