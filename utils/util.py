# ============================================
# utils/util.py
# 共通ユーティリティ（日時・数値処理）
# ============================================

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import math
from typing import Optional


# --------------------------------------------
# JST 時刻
# --------------------------------------------
JST = timezone(timedelta(hours=9))


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_now_str() -> str:
    return jst_now().strftime("%Y-%m-%d %H:%M")


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


# --------------------------------------------
# 安全な数値処理
# --------------------------------------------
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or b is None:
            return default
        return a / b
    except Exception:
        return default


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


# --------------------------------------------
# RR 下限（地合い連動）
# --------------------------------------------
def rr_min_by_market(market_score: float) -> float:
    """
    地合いが弱いほど RR を要求する
    """
    if market_score >= 75:
        return 1.8
    if market_score >= 65:
        return 2.0
    if market_score >= 55:
        return 2.2
    if market_score >= 45:
        return 2.4
    return 2.6


# --------------------------------------------
# レバレッジ目安
# --------------------------------------------
def leverage_by_market(market_score: float, macro_risk: bool) -> float:
    if macro_risk:
        return 1.0

    if market_score >= 75:
        return 2.0
    if market_score >= 65:
        return 1.7
    if market_score >= 55:
        return 1.3
    return 1.0


# --------------------------------------------
# ExpectedDays 推定
# --------------------------------------------
def estimate_days(tp2: float, entry: float, atr: float) -> float:
    """
    TP2 到達までの想定日数（速度評価用）
    """
    if atr <= 0:
        return 99.0
    return abs(tp2 - entry) / atr


# --------------------------------------------
# R/day
# --------------------------------------------
def calc_r_per_day(rr: float, days: float) -> float:
    if days <= 0:
        return 0.0
    return rr / days


def round_price(x: float, ndigits: int = 1) -> float:
    try:
        return round(float(x), ndigits)
    except Exception:
        return float("nan")


def to_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default