from __future__ import annotations

from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today():
    return jst_now().date()


def jst_today_str():
    return jst_now().strftime("%Y-%m-%d")