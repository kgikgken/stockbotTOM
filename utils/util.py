from __future__ import annotations
from datetime import datetime, timezone, timedelta


def jst_now() -> datetime:
    """
    現在時刻を JST で返す。
    """
    return datetime.now(timezone(timedelta(hours=9)))


def jst_today_str() -> str:
    """
    今日の日付(YYYY-MM-DD) を JST 基準で返す。
    例: "2025-12-08"
    """
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    """
    今日の日付を date 型で返す。
    """
    return jst_now().date()