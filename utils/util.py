# ============================================
# utils/util.py
# 共通ユーティリティ（日時・数値・リスク計算）
# ============================================

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import math


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
# 地合い連動 RR 下限
# --------------------------------------------
def rr_min_by_market(market_score: float) -> float:
    """
    地合いが悪いほど、より高い RR を要求する
    """
    if market_score >= 70:
        return 1.8
    elif market_score >= 60:
        return 2.0
    elif market_score >= 50:
        return 2.2
    else:
        return 2.5


# --------------------------------------------
# レバレッジ目安（マクロ考慮）
# --------------------------------------------
def leverage_by_market(market_score: float, macro_risk: bool) -> float:
    """
    マクロ警戒時は無条件で 1.0 倍
    """
    if macro_risk:
        return 1.0

    if market_score >= 75:
        return 2.0
    elif market_score >= 65:
        return 1.7
    elif market_score >= 55:
        return 1.3
    else:
        return 1.0


# --------------------------------------------
# Expected Days（速度評価）
# --------------------------------------------
def estimate_days(entry: float, tp2: float, atr: float) -> float:
    """
    TP2 到達までの想定日数
    """
    if atr <= 0:
        return 99.0
    return abs(tp2 - entry) / atr


# --------------------------------------------
# R / day
# --------------------------------------------
def calc_r_per_day(rr: float, days: float) -> float:
    if days <= 0:
        return 0.0
    return rr / days


# --------------------------------------------
# ロット事故チェック
# --------------------------------------------
def calc_max_loss(
    entry: float,
    stop: float,
    quantity: float,
) -> float:
    """
    想定最大損失（金額）
    """
    if entry <= 0 or quantity <= 0:
        return 0.0
    return abs(entry - stop) * quantity


def lot_risk_ratio(
    max_loss: float,
    total_asset: float,
) -> float:
    """
    総資産に対する最大損失比率
    """
    if total_asset <= 0:
        return 0.0
    return max_loss / total_asset