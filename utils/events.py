# ============================================
# utils/events.py
# 重要イベント（FOMC/日銀/CPI/雇用統計など）読み込み＆判定
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import csv
import os

from utils.util import JST


@dataclass(frozen=True)
class MacroEvent:
    name: str
    dt_jst: datetime


def _parse_dt_jst(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    # 例: "2026-01-05 00:00"
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m-%d":
                dt = dt.replace(hour=0, minute=0, second=0)
            return dt.replace(tzinfo=JST)
        except Exception:
            pass
    return None


def load_events_csv(path: str = "events.csv") -> List[MacroEvent]:
    if not os.path.exists(path):
        return []

    out: List[MacroEvent] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or row.get("event") or "").strip()
            dt = _parse_dt_jst(row.get("datetime") or row.get("dt") or row.get("date") or "")
            if name and dt:
                out.append(MacroEvent(name=name, dt_jst=dt))

    out.sort(key=lambda x: x.dt_jst)
    return out


def nearest_event(events: List[MacroEvent], now_jst: datetime) -> Optional[Tuple[MacroEvent, int]]:
    """
    直近イベントと、今から何日後か（整数）を返す
    """
    future = [e for e in events if e.dt_jst >= now_jst]
    if not future:
        return None
    ev = future[0]
    delta_days = (ev.dt_jst.date() - now_jst.date()).days
    return ev, delta_days


def macro_risk_on(events: List[MacroEvent], now_jst: datetime, days_ahead: int = 2) -> bool:
    """
    イベント接近の警戒フラグ
    """
    x = nearest_event(events, now_jst)
    if not x:
        return False
    _, d = x
    return d <= days_ahead