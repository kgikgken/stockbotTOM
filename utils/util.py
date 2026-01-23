from __future__ import annotations

import math
import datetime as _dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


def jst_now() -> _dt.datetime:
    return _dt.datetime.now(tz=JST)


def jst_today_date() -> _dt.date:
    return jst_now().date()


def jst_today_str() -> str:
    return jst_today_date().isoformat()


def safe_float(x, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (float, int)):
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fmt_yen(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{int(round(x)):,} 円"


def fmt_ratio(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def rsi14(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ma_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr14(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def atr_pct_last(df: pd.DataFrame) -> float:
    atr = atr14(df).iloc[-1]
    px = df["Close"].iloc[-1]
    if not np.isfinite(atr) or px <= 0:
        return float("nan")
    return float(atr / px)


def adv20(df: pd.DataFrame) -> float:
    v = df["Volume"].tail(20).mean()
    c = df["Close"].tail(20).mean()
    if not np.isfinite(v) or not np.isfinite(c):
        return float("nan")
    return float(v * c)


def pct_change(a: float, b: float) -> float:
    if a == 0 or not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float((b - a) / a)
