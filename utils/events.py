from __future__ import annotations

import os
from datetime import date
from typing import List, Dict, Optional

import pandas as pd

from utils.util import parse_event_datetime_jst


def load_events(path: str) -> List[Dict[str, str]]:
    if not path or (not os.path.exists(path)):
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


def build_event_section(events: List[Dict[str, str]], today_date: date) -> List[str]:
    """
    LINE出力用：前日〜2日後だけ表示。無ければ「特になし」
    """
    lines: List[str] = []
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
            lines.append(f"⚠ {ev['label']}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")

    if not lines:
        return ["- 特になし"]
    return lines


def detect_macro_caution(events: List[Dict[str, str]], today_date: date) -> bool:
    """
    “イベント接近”の判定：
    0日後 or 1日後 に重要イベントがあるなら ON（新規禁止・候補2に制限）
    """
    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if delta in (0, 1):
            return True
    return False