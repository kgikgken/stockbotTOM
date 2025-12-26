# utils/util.py
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

def jst_today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")

def jst_today_date():
    return datetime.now(JST).date()

def parse_event_datetime_jst(dt, d, t):
    try:
        if dt:
            return datetime.strptime(dt, "%Y-%m-%d %H:%M").replace(tzinfo=JST)
        if d and t:
            return datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=JST)
        if d:
            return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=JST)
    except Exception:
        return None
    return None