# ============================================
# utils/util.py
# 共通ユーティリティ（日時・数値・地合い連動）
# ============================================

from datetime import datetime, timezone, timedelta
import math

# --------------------------------------------
# JST 時刻
# --------------------------------------------
JST = timezone(timedelta(hours=9))


def jst_now():
    return datetime.now(JST)


def jst_now_str():
    return jst_now().strftime("%Y-%m-%d %H:%M")


def jst_today_str():
    return jst_now().strftime("%Y-%m-%d")


# --------------------------------------------
# 安全な数値処理
# --------------------------------------------
def safe_div(a, b, default=0.0):
    try:
        if b == 0 or b is None:
            return default
        return a / b
    except Exception:
        return default


def clamp(x, low, high):
    try:
        return max(low, min(high, x))
    except Exception:
        return low


# --------------------------------------------
# RR 下限（地合い連動）
# --------------------------------------------
def rr_min_by_market(market_score: float) -> float:
    """
    地合いが弱いほど RR を要求する（完全固定禁止）
    """
    if market_score >= 75:
        return 1.8
    elif market_score >= 65:
        return 2.0
    elif market_score >= 55:
        return 2.2
    elif market_score >= 45:
        return 2.4
    else:
        return 2.6


# --------------------------------------------
# レバレッジ目安（マクロ考慮）
# --------------------------------------------
def leverage_by_market(market_score: float, macro_risk: bool) -> float:
    """
    マクロ警戒時は無条件で 1.0
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
# 最大建玉目安
# --------------------------------------------
def calc_max_position(capital: float, leverage: float) -> float:
    try:
        return capital * leverage
    except Exception:
        return 0.0


# --------------------------------------------
# ExpectedDays 推定
# --------------------------------------------
def estimate_days(tp2: float, entry: float, atr: float) -> float:
    """
    TP2 到達までの想定日数（速度評価用）
    """
    try:
        if atr <= 0:
            return 99.0
        return abs(tp2 - entry) / atr
    except Exception:
        return 99.0


# --------------------------------------------
# R/day
# --------------------------------------------
def calc_r_per_day(rr: float, days: float) -> float:
    try:
        if days <= 0:
            return 0.0
        return rr / days
    except Exception:
        return 0.0


# --------------------------------------------
# RR 分布の自然化（固定回避）
# --------------------------------------------
def normalize_rr(rr: float) -> float:
    """
    RR が 3.00 に固まりすぎるのを防ぐための微調整
    """
    try:
        # 不自然な固定値を丸め直す
        return round(rr, 2)
    except Exception:
        return rr


# --------------------------------------------
# AdjEV 判定
# --------------------------------------------
def is_valid_adjev(adjev: float, threshold: float = 0.5) -> bool:
    """
    AdjEV 下限フィルタ
    """
    try:
        return adjev >= threshold
    except Exception:
        return False


# --------------------------------------------
# 週次制限チェック
# --------------------------------------------
def weekly_limit_ok(current: int, limit: int) -> bool:
    try:
        return current < limit
    except Exception:
        return False