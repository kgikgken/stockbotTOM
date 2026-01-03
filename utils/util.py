# utils/util.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterable, List, Any

import math

# ============================================================
# Time (JST)
# ============================================================
JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    return jst_now().date()


def parse_event_datetime_jst(
    dt_str: Optional[str],
    date_str: Optional[str],
    time_str: Optional[str],
) -> Optional[datetime]:
    """
    events.csv で以下を許容して JST datetime を返す
      - datetime: "2025-12-11 03:00" / "2025-12-11 03:00:00"
      - date: "2025-12-11" と time:"03:00" / "03:00:00"
      - date: "2025-12-11" のみ（00:00扱い）
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


# ============================================================
# Safe numeric helpers
# ============================================================
def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def clamp(x: float, lo: float, hi: float) -> float:
    if not is_finite(x):
        return lo
    return float(min(max(float(x), float(lo)), float(hi)))


# ============================================================
# Text helpers (LINE chunk)
# ============================================================
def chunk_text(text: str, chunk_size: int = 3800) -> List[str]:
    """
    LINE Worker 送信は長文で落ちる可能性があるので分割する前提。
    chunk_size は安全側（4000未満推奨）
    """
    if not text:
        return [""]
    if chunk_size <= 0:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# ============================================================
# Simple labels
# ============================================================
@dataclass(frozen=True)
class Action:
    EXEC_NOW: str = "即IN可"
    LIMIT_WAIT: str = "指値待ち"
    WATCH_ONLY: str = "監視のみ"


def yesno(flag: bool) -> str:
    return "Y" if bool(flag) else "N"