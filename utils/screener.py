# ============================================
# utils/screener.py
# 銘柄スクリーニング中核ロジック
# ============================================

from __future__ import annotations

from typing import List, Dict

from utils.rr_ev import (
    calc_rr,
    estimate_pwin,
    calc_ev,
    apply_market_adjustment,
    is_rr_valid,
    evaluate_speed,
)
from utils.util import clamp


# --------------------------------------------
# スクリーニング本体
# --------------------------------------------
def screen_stocks(
    candidates: List[Dict],
    market_score: float,
    delta_3d: float,
    macro_risk: bool,
    max_candidates: int,
) -> List[Dict]:
    """
    各銘柄の RR / EV / AdjEV / R/day を計算し、
    条件を満たすものだけを返す
    """

    results = []

    for c in candidates:
        entry = c["entry"]
        stop = c["stop"]
        tp2 = c["tp2"]
        atr = c["atr"]

        # RR
        rr = calc_rr(entry, stop, tp2)
        if not is_rr_valid(rr, market_score):
            continue

        # 勝率代理
        pwin = estimate_pwin(c.get("features", {}))

        # EV
        ev = calc_ev(rr, pwin)

        # 地合い補正
        adj_ev = apply_market_adjustment(
            ev,
            market_score,
            delta_3d,
            macro_risk,
        )

        # EV 足切り
        if adj_ev < 0.5:
            continue

        # 速度評価
        days, r_day = evaluate_speed(entry, tp2, atr, rr)

        # 速度足切り
        if days > 7 or r_day < 0.5:
            continue

        c2 = dict(c)
        c2.update(
            {
                "rr": round(rr, 2),
                "pwin": round(pwin, 2),
                "ev": round(ev, 2),
                "adj_ev": round(adj_ev, 2),
                "expected_days": round(days, 1),
                "r_per_day": round(r_day, 2),
            }
        )

        results.append(c2)

    # AdjEV → R/day 優先で並び替え
    results.sort(
        key=lambda x: (x["adj_ev"], x["r_per_day"]),
        reverse=True,
    )

    return results[:max_candidates]