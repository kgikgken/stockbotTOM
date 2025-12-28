from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta


EVENTS_PATH = "events.csv"   # label,date,type を想定
EARNINGS_COL = "earnings_date"


def load_events() -> pd.DataFrame:
    try:
        return pd.read_csv(EVENTS_PATH)
    except Exception:
        return pd.DataFrame()


def is_event_near(date: datetime.date, window: int = 1) -> bool:
    df = load_events()
    if df.empty:
        return False

    df["date"] = pd.to_datetime(df["date"]).dt.date
    for d in df["date"]:
        if abs((d - date).days) <= window:
            return True
    return False


def earnings_filter(universe: pd.DataFrame, today: datetime.date) -> pd.DataFrame:
    if EARNINGS_COL not in universe.columns:
        return universe

    uni = universe.copy()
    uni[EARNINGS_COL] = pd.to_datetime(uni[EARNINGS_COL], errors="coerce").dt.date

    mask = uni[EARNINGS_COL].isna() | (
        (uni[EARNINGS_COL] - today).abs().dt.days > 3
    )
    return uni[mask]