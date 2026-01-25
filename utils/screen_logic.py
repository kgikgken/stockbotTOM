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

def no_trade_conditions(market_score: float, delta3: float, macro_warn: bool) -> Tuple[bool, Optional[str]]:
    """NO-TRADE is an *emergency brake* only.

    MarketScore is primarily for exit-speed control (not a gate). Therefore,
    this function triggers NO-TRADE only in extreme regimes.
    """
    if macro_warn and market_score < 40:
        return True, "重要イベント×低地合いのため新規停止"

    # Extreme risk-off
    if market_score < 30:
        return True, "地合いが極端に悪いため新規停止"

    if market_score < 40 and delta3 <= -10:
        return True, "地合い悪化が急なため新規停止"

    return False, None



