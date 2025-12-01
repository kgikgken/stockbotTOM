from __future__ import annotations

from datetime import datetime, timedelta, timezone


JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    """今日の日付を 'YYYY-MM-DD' で返す（JST）"""
    return jst_now().strftime("%Y-%m-%d")