from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Tuple

import pandas as pd


MACRO_KEYWORDS = ("fomc", "cpi", "boj", "日銀", "雇用", "gdp", "pce", "sq", "決算集中")


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except Exception:
        try:
            return pd.to_datetime(text).date()
        except Exception:
            return None


def build_event_section(today: date, csv_path: str | Path = "events.csv") -> Tuple[List[str], bool]:
    path = Path(csv_path)
    if not path.exists():
        return [], False
    try:
        df = pd.read_csv(path)
    except Exception:
        return [], False
    if df.empty:
        return [], False

    out: List[str] = []
    macro_on = False
    rows: list[tuple[int, date, str]] = []
    for _, row in df.iterrows():
        d = _parse_date(row.get("date"))
        if d is None:
            continue
        title = str(row.get("title", row.get("event", ""))).strip()
        if not title:
            continue
        importance = int(float(row.get("importance", 3) or 3))
        watch_days = int(float(row.get("watch_days", 1) or 1))
        delta = (d - today).days
        if -1 <= delta <= max(3, watch_days):
            rows.append((importance, d, title))
        is_macro = importance >= 4 or any(k in title.lower() for k in MACRO_KEYWORDS)
        if is_macro and -1 <= delta <= 1:
            macro_on = True

    rows.sort(key=lambda x: (x[1], -x[0], x[2]))
    for importance, d, title in rows[:6]:
        out.append(f"{d.isoformat()} | I{importance} | {title}")
    return out, macro_on
