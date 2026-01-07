# ============================================
# utils/rr_ev.py
# RR / EV / 補正EV / 速度評価
# ============================================

from __future__ import annotations

from utils.util import (
    safe_div,
    rr_min_by_market,
    estimate_days,
    calc_r_per_day,
)


# --------------------------------------------
# RR 計算
# --------------------------------------------
def calc_rr(entry: float, stop: float, tp2: float) -> float:
    """
    RR = (TP2 - Entry) / (Entry - Stop)
    """
    risk = entry - stop
    reward = tp2 - entry
    return safe_div(reward, risk, 0.0)


# --------------------------------------------
# 勝率代理（Pwin）
# --------------------------------------------
def estimate_pwin(features: dict) -> float:
    """
    厳密な勝率は推定不可なので、
    特徴量から 0〜1 に正規化した代理値を作る
    """
    score = 0.0
    score += features.get("trend_strength", 0.0) * 0.30
    score += features.get("relative_strength", 0.0) * 0.25
    score += features.get("sector_rank", 0.0) * 0.15
    score += features.get("volume_quality", 0.0) * 0.15
    score -= features.get("gap_risk", 0.0) * 0.25

    # clamp
    return max(0.05, min(0.85, score))


# --------------------------------------------
# EV 計算
# --------------------------------------------
def calc_ev(rr: float, pwin: float) -> float:
    """
    EV = Pwin * RR - (1 - Pwin) * 1R
    """
    return pwin * rr - (1 - pwin) * 1.0


# --------------------------------------------
# 地合い補正
# --------------------------------------------
def apply_market_adjustment(
    ev: float,
    market_score: float,
    delta_3d: float,
    macro_risk: bool,
) -> float:
    multiplier = 1.0

    if market_score >= 70 and delta_3d >= 0:
        multiplier = 1.05
    elif delta_3d <= -5:
        multiplier = 0.70

    if macro_risk:
        multiplier *= 0.75

    return ev * multiplier


# --------------------------------------------
# RR 足切り判定（地合い連動）
# --------------------------------------------
def is_rr_valid(rr: float, market_score: float) -> bool:
    rr_min = rr_min_by_market(market_score)
    return rr >= rr_min


# --------------------------------------------
# 速度評価
# --------------------------------------------
def evaluate_speed(entry: float, tp2: float, atr: float, rr: float):
    """
    ExpectedDays と R/day を返す
    """
    days = estimate_days(tp2, entry, atr)
    r_day = calc_r_per_day(rr, days)
    return days, r_day