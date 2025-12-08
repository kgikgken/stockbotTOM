from __future__ import annotations
from datetime import datetime, timedelta, timezone


# ============================================================
# JST utilities
# ============================================================

_JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    """
    現在のJST日時
    """
    return datetime.now(_JST)


def jst_today_date():
    """
    JST今日の日付 (datetime.date)
    """
    return jst_now().date()


def jst_today_str() -> str:
    """
    JST今日の日付 "YYYY-MM-DD"
    """
    return jst_now().strftime("%Y-%m-%d")


def jst_now_str() -> str:
    """
    JST現在の日時 "YYYY-MM-DD HH:MM:SS"
    """
    return jst_now().strftime("%Y-%m-%d %H:%M:%S")