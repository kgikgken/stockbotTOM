# utils/events.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from utils.util import DEFAULT_TZ, EventState


@dataclass(frozen=True)
class MacroEvent:
    name: str
    when_jst: datetime


def _parse_dt(s: str) -> Optional[datetime]:
    try:
        # "YYYY-MM-DD HH:MM" または "YYYY-MM-DD"
        s = str(s).strip()
        if not s:
            return None
        if len(s) == 10:
            dt = datetime.strptime(s, "%Y-%m-%d")
            return dt.replace(tzinfo=DEFAULT_TZ)
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=DEFAULT_TZ)
    except Exception:
        return None


def load_events(path: str) -> EventState:
    """
    events.csv:
      - name
      - when_jst (YYYY-MM-DD HH:MM)
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return EventState(macro_event_near=False, macro_event_text="なし")

    events: List[MacroEvent] = []
    for _, r in df.iterrows():
        name = str(r.get("name", "")).strip()
        when = _parse_dt(str(r.get("when_jst", "")).strip())
        if name and when:
            events.append(MacroEvent(name=name, when_jst=when))

    now = datetime.now(DEFAULT_TZ)
    # 「接近」の定義：48時間以内に未来イベントがある
    near_list = []
    for e in events:
        if e.when_jst >= now and e.when_jst <= now + timedelta(hours=48):
            near_list.append(e)

    if not near_list:
        return EventState(macro_event_near=False, macro_event_text="なし")

    # 一番近いイベントだけ表示
    near_list.sort(key=lambda x: x.when_jst)
    e = near_list[0]
    delta = e.when_jst - now
    days = int(delta.total_seconds() // 86400)
    hours = int((delta.total_seconds() % 86400) // 3600)

    text = f"⚠ {e.name}（{e.when_jst.strftime('%Y-%m-%d %H:%M')} JST / {days}日{hours}時間後）"
    return EventState(macro_event_near=True, macro_event_text=text)