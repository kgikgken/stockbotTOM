# utils/util.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

DEFAULT_TZ = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(DEFAULT_TZ)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fmt_int(n: float) -> str:
    try:
        return f"{int(round(n)):,}"
    except Exception:
        return "n/a"


def fmt_float(x: float, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def pct(x: float, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except Exception:
        return "n/a"


@dataclass(frozen=True)
class MarketState:
    score: float
    delta_3d: float
    regime: str  # "bull" / "neutral" / "bear"
    macro_caution: bool
    leverage: float
    max_gross: float
    no_trade: bool
    reason: str


@dataclass(frozen=True)
class EventState:
    macro_event_near: bool
    macro_event_text: str


def dict_get(d: Dict[str, Any], k: str, default: Any = None) -> Any:
    try:
        return d.get(k, default)
    except Exception:
        return default