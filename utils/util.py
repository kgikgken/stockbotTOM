from datetime import datetime, timedelta, timezone
from typing import Optional

JST = timezone(timedelta(hours=9))

def jst_today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")

def jst_today_date():
    return datetime.now(JST).date()

def parse_event_datetime_jst(dt_str, date_str, time_str) -> Optional[datetime]:
    if dt_str:
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=JST)
        except Exception:
            pass
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=JST)
        except Exception:
            pass
    return None