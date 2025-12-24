from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    return jst_now().date()


def parse_event_datetime_jst(dt_str: str | None, date_str: str | None, time_str: str | None) -> Optional[datetime]:
    """
    events.csv の入力を柔軟に解釈して JST datetime を返す。
      - datetime: "2025-12-11 03:00"
      - date: "2025-12-11" と time:"03:00"
      - date: "2025-12-11" のみ（00:00）
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
