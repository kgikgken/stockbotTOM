from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    return jst_now().date()


def safe_float(x, default=float("nan")) -> float:
    try:
        v = float(x)
        if v != v:  # nan
            return float(default)
        if v == float("inf") or v == float("-inf"):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def clamp(v: float, lo: float, hi: float) -> float:
    if v != v:
        return v
    return float(max(lo, min(hi, v)))


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


@dataclass(frozen=True)
class Config:
    # スイング
    SWING_MAX_FINAL: int = 5
    WATCH_MAX: int = 10

    # 決算
    EARNINGS_EXCLUDE_DAYS: int = 3

    # Universe足切り（トレード候補のみ）
    PRICE_MIN: float = 200.0
    PRICE_MAX: float = 15000.0
    ADV20_MIN_JPY: float = 200_000_000.0  # 200M（最低100Mは許容するが監視寄り）
    ADV20_WARN_JPY: float = 100_000_000.0
    ATRPCT_MIN: float = 0.015            # 1.5%
    ATRPCT_MAX: float = 0.06             # 6% 事故ゾーン
    STOPLIM_HIT_60D_EXCLUDE: int = 2      # 任意。データが無ければ無視

    # 地合いNO-TRADE
    NOTRADE_SCORE_LT: int = 45
    NOTRADE_DELTA3D_LE: int = -5
    NOTRADE_SCORE_LT_2: int = 55

    # RR/EV/速度
    RR_MIN: float = 1.8
    EV_MIN_R: float = 0.30
    EV_MIN_R_NEUTRAL: float = 0.40
    EXPECTED_DAYS_MAX: float = 5.0
    RPDAY_MIN: float = 0.50

    # 分散
    MAX_SAME_SECTOR: int = 2
    CORR_MAX: float = 0.75

    # リスク管理（ロット事故）
    RISK_PER_TRADE: float = 0.015
    MAX_PORTFOLIO_RISK_PCT: float = 0.05  # 5%超なら警告

    # セクター採用
    SECTOR_TOP_N: int = 5