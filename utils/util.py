from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
from typing import Optional

import pandas as pd

JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date() -> date:
    return jst_now().date()


def parse_event_datetime_jst(dt_str: str | None, date_str: str | None, time_str: str | None) -> Optional[datetime]:
    """
    events.csv supports:
      - datetime: "2025-12-11 03:00" (JST assumed)
      - date: "2025-12-11" + time:"03:00"
      - date: "2025-12-11" only (00:00)
    """
    dt_str = (dt_str or "").strip()
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()

    if dt_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_str, fmt).replace(tzinfo=JST)
            except Exception:
                pass

    if date_str and time_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(f"{date_str} {time_str}", fmt).replace(tzinfo=JST)
            except Exception:
                pass

    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=JST)
        except Exception:
            return None

    return None


def business_days_between(a: date, b: date) -> int:
    """
    Signed business-day difference: b - a in business days.
    """
    try:
        if a == b:
            return 0
        if a < b:
            rng = pd.bdate_range(a, b)
            return int(len(rng) - 1)
        else:
            rng = pd.bdate_range(b, a)
            return -int(len(rng) - 1)
    except Exception:
        return (b - a).days
