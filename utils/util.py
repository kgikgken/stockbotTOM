from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterable, List, Tuple, Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf

JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    return jst_now().date()


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def chunk_text(text: str, chunk_size: int = 3800) -> List[str]:
    if not text:
        return [""]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def parse_event_datetime_jst(dt_str: str | None, date_str: str | None, time_str: str | None) -> Optional[datetime]:
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


def yf_history(
    ticker: str,
    period: str = "260d",
    interval: str = "1d",
    auto_adjust: bool = True,
    tries: int = 2,
    sleep: float = 0.4,
) -> Optional[pd.DataFrame]:
    for _ in range(max(1, tries)):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=auto_adjust)
            if df is not None and not df.empty:
                df = df.copy()
                for c in ("Open", "High", "Low", "Close"):
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                if "Volume" in df.columns:
                    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
                return df
        except Exception:
            time.sleep(sleep)
    return None


def read_csv_safely(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def pick_ticker_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("ticker", "code", "Ticker", "Code"):
        if col in df.columns:
            return col
    return None


def pick_sector_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("sector", "industry_big", "Sector", "Industry"):
        if col in df.columns:
            return col
    return None