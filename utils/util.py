from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
from typing import Optional

JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date() -> date:
    return jst_now().date()


def safe_float(x, default=float("nan")) -> float:
    try:
        v = float(x)
        if v != v:  # nan
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def parse_event_datetime_jst(dt_str: str | None, date_str: str | None, time_str: str | None) -> Optional[datetime]:
    """
    events.csv で
      - datetime: "2026-01-05 00:00"
      - date: "2026-01-05" と time:"00:00"
      - date: "2026-01-05" のみ
    を許容して JST datetime を返す
    """
    dt_str = (dt_str or "").strip()
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()

    # 1) datetime優先
    if dt_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_str, fmt).replace(tzinfo=JST)
            except Exception:
                pass

    # 2) date + time
    if date_str and time_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(f"{date_str} {time_str}", fmt).replace(tzinfo=JST)
            except Exception:
                pass

    # 3) dateのみ（00:00）
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=JST)
        except Exception:
            return None

    return None


def weekday_monday(d: date) -> date:
    # 月曜起点
    return d - timedelta(days=d.weekday())