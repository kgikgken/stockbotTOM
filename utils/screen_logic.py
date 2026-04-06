from __future__ import annotations

from utils.util import env_float, env_int


def weekly_max_new() -> int:
    return env_int("WEEKLY_MAX_NEW", 3)


def max_display(lane: str = "trend") -> int:
    if lane == "leaders":
        return env_int("LEADERS_MAX_DISPLAY", 6)
    return env_int("TREND_MAX_DISPLAY", 6)


def no_trade_conditions(mkt_score: int, delta3: int, macro_warn: bool = False) -> bool:
    if macro_warn and env_int("MACRO_FORCE_NO_TRADE", 1) == 1:
        return True
    if int(mkt_score) <= env_int("NO_TRADE_MKT_SCORE_MAX", 35):
        return True
    if int(delta3) <= env_int("NO_TRADE_DELTA3_MAX", -20):
        return True
    return False


def rs_pct_min_by_market(mkt_score: int) -> int:
    s = int(mkt_score)
    if s >= 70:
        return env_int("TREND_RS_PCT_MIN_STRONG", 50)
    if s >= 60:
        return env_int("TREND_RS_PCT_MIN_NORMAL", 55)
    if s >= 50:
        return env_int("TREND_RS_PCT_MIN_SOFT", 60)
    return env_int("TREND_RS_PCT_MIN_WEAK", 70)


def rs_comp_min_by_market(mkt_score: int) -> float:
    s = int(mkt_score)
    if s >= 70:
        return env_float("TREND_RS_COMP_MIN_STRONG", -0.5)
    if s >= 60:
        return env_float("TREND_RS_COMP_MIN_NORMAL", 0.0)
    if s >= 50:
        return env_float("TREND_RS_COMP_MIN_SOFT", 0.8)
    return env_float("TREND_RS_COMP_MIN_WEAK", 1.5)
