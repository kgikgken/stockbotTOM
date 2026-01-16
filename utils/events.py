from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import csv

from utils.util import parse_event_datetime_jst, JST


@dataclass(frozen=True)
class MacroEvent:
    name: str
    dt_jst: str  # "YYYY-mm-dd HH:MM"
    days_until: int


IMPORTANT_KEYWORDS = (
    "CPI",
    "FOMC",
    "雇用統計",
    "日銀",
    "BOJ",
    "GDP",
    "PCE",
    "CPI",
)


def load_macro_events(events_path: str, today: date, lookahead_days: int = 3) -> List[MacroEvent]:
    """events.csv から重要イベントを抽出。

    想定カラム例：
      - name
      - datetime (or date/time)

    形式が違っても、name/date/time/datetime を探して読む。
    """
    out: List[MacroEvent] = []
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or row.get("event") or row.get("title") or "").strip()
                if not name:
                    continue
                if not any(k in name for k in IMPORTANT_KEYWORDS):
                    continue

                dt = parse_event_datetime_jst(
                    row.get("datetime"),
                    row.get("date"),
                    row.get("time"),
                )
                if not dt:
                    continue

                days_until = (dt.date() - today).days
                if -1 <= days_until <= lookahead_days:
                    out.append(
                        MacroEvent(
                            name=name,
                            dt_jst=dt.astimezone(JST).strftime("%Y-%m-%d %H:%M"),
                            days_until=days_until,
                        )
                    )
    except FileNotFoundError:
        return []
    except Exception:
        return []

    out.sort(key=lambda e: (e.days_until, e.dt_jst, e.name))
    return out


def macro_alert_on(events: List[MacroEvent]) -> bool:
    return len(events) > 0
