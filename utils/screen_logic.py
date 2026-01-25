from __future__ import annotations

# ------------------------------------------------------------
# フィルタ・制御の仕様（最新版）
# ------------------------------------------------------------
# - MarketScore はゲートではない（撤退速度制御専用）
# - RR下限は固定（1.8）
# - 回転効率（R/日）下限は Setup 別
# - 表示数は最大5
# - 週次新規枠は3（既存運用に合わせる）
# ------------------------------------------------------------

def rr_min_by_market(mkt_score: int) -> float:
    # 固定（MarketScoreで変えない）
    return 1.8

def rday_min_by_setup(setup: str) -> float:
    # 回転効率（R/日）下限（Setup別）
    if setup == "A1-Strong":
        return 0.50
    if setup == "A1":
        return 0.45
    if setup == "A2":
        return 0.55
    if setup == "B":
        return 0.60
    if setup == "S":
        return 0.70
    return 0.45

def max_display(macro_on: bool) -> int:
    return 5

def weekly_max_new() -> int:
    return 3

def no_trade_conditions(mkt_score: int, delta3: float) -> bool:
    # NO-TRADE 前提（安全装置）
    # ※ MarketScoreは撤退制御が主だが、極端に悪い局面では新規停止
    if mkt_score < 45:
        return True
    if delta3 <= -5.0 and mkt_score < 55:
        return True
    return False
