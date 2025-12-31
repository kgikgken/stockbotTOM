from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    return jst_now().date()


def safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def pct(x: float) -> str:
    return f"{x:+.2f}%"


def fmt_yen(x: float) -> str:
    try:
        return f"{int(round(x)):,}円"
    except Exception:
        return "n/a"