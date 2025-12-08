from __future__ import annotations
from datetime import datetime, timedelta, timezone

# JST タイムゾーン
JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    """JST 現在時刻（datetime）"""
    return datetime.now(JST)


def jst_today_str() -> str:
    """JST 今日の日付を 'YYYY-MM-DD' で返す"""
    return jst_now().strftime("%Y-%m-%d")