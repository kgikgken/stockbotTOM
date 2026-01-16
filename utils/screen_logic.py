from __future__ import annotations

def rr_min_by_market(mkt_score: int) -> float:
    if mkt_score >= 70:
        return 1.8
    if mkt_score >= 60:
        return 2.0
    if mkt_score >= 50:
        return 2.2
    return 2.5

def rday_min_by_setup(setup: str) -> float:
    if setup == "A1-Strong":
        return 0.45
    if setup == "A1":
        return 0.50
    if setup == "A2":
        return 0.55
    if setup == "B":
        return 0.65
    return 0.50

def max_display(macro_on: bool) -> int:
    return 5

def weekly_max_new() -> int:
    return 3

def no_trade_conditions(mkt_score: int, delta3: float) -> bool:
    if mkt_score < 45:
        return True
    if delta3 <= -5.0 and mkt_score < 55:
        return True
    return False
