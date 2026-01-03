# utils/rr_ev.py
from __future__ import annotations

import numpy as np
from typing import Dict

# ============================================================
# Regime Multiplier（地合い補正）
# ============================================================
def calc_regime_multiplier(
    market_score: int,
    delta_3d: float,
    has_event_risk: bool = False
) -> float:
    """
    EVを現実値に落とすための環境補正
    """
    mult = 1.0

    # 強地合い
    if market_score >= 60 and delta_3d >= 0:
        mult *= 1.05

    # 崩れ初動
    if delta_3d <= -5:
        mult *= 0.70

    # マクロイベント警戒
    if has_event_risk:
        mult *= 0.75

    return float(mult)


# ============================================================
# RR / EV / 速度 の最終判定
# ============================================================
def final_rr_ev_judge(
    rr: float,
    ev: float,
    adj_ev: float,
    expected_days: float,
    r_per_day: float,
    *,
    min_rr: float = 2.2,
    min_ev_r: float = 0.4,
    min_r_per_day: float = 0.5,
    max_days: float = 5.0,
) -> Dict:
    """
    採否を完全機械化
    """
    reasons = []

    if not np.isfinite(rr) or rr < min_rr:
        reasons.append("RR不足")

    if not np.isfinite(ev) or ev < min_ev_r * rr:
        reasons.append("EV不足")

    if not np.isfinite(expected_days) or expected_days > max_days:
        reasons.append("遅すぎ")

    if not np.isfinite(r_per_day) or r_per_day < min_r_per_day:
        reasons.append("速度不足")

    passed = len(reasons) == 0

    return {
        "passed": passed,
        "reject_reasons": reasons,
        "rr": float(rr),
        "ev": float(ev),
        "adj_ev": float(adj_ev),
        "expected_days": float(expected_days),
        "r_per_day": float(r_per_day),
    }


# ============================================================
# NO-TRADE DAY 判定
# ============================================================
def is_no_trade_day(
    market_score: int,
    delta_3d: float,
    avg_adj_ev: float,
    gu_ratio: float
) -> Dict:
    """
    新規ゼロを強制する日
    """
    reasons = []

    if market_score < 45:
        reasons.append("MarketScore<45")

    if delta_3d <= -5 and market_score < 55:
        reasons.append("崩れ初動")

    if avg_adj_ev < 0.3:
        reasons.append("平均AdjEV不足")

    if gu_ratio >= 0.6:
        reasons.append("GU比率過多")

    return {
        "no_trade": len(reasons) > 0,
        "reasons": reasons
    }