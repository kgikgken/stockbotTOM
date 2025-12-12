from __future__ import annotations

from datetime import datetime, timezone, timedelta


def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    return jst_now().date()