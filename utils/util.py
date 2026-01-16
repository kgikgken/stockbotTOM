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


def parse_event_datetime_jst(dt_str: str | None, date_str: str | None, time_str: str | None) -> Optional[datetime]:
    """events.csv を JST datetime に変換。

    許容：
      - datetime: "YYYY-mm-dd HH:MM" or "YYYY-mm-dd HH:MM:SS"
      - date + time
      - date only (00:00)
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


def fmt_yen(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)


def fmt_pct(x: float, digits: int = 2) -> str:
    try:
        return f"{float(x):+.{digits}f}%"
    except Exception:
        return str(x)


def safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default
