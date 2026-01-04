from __future__ import annotations
import math, os
from datetime import datetime, timedelta, timezone
from typing import Any
import pandas as pd

JST = timezone(timedelta(hours=9))

def jst_now() -> datetime:
    return datetime.now(tz=JST)

def jst_today_date():
    return jst_now().date()

def safe_print(msg: str) -> None:
    try:
        print(msg)
    except Exception:
        print(str(msg).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def parse_date_yyyy_mm_dd(s: Any):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    try:
        return pd.to_datetime(str(s)).date()
    except Exception:
        return None