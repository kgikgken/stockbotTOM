from __future__ import annotations

from typing import Dict, Tuple

# MarketScoreは撤退速度制御専用。銘柄選別のゲートにはしない。
# フィルターは固定条件、表示順は CAGR寄与度（pt）降順。

def rr_min_fixed() -> float:
    return 1.8

def ev_min_fixed() -> float:
    return 0.50

def rotation_min_by_setup() -> Dict[str, float]:
    return {
        "A1-Strong": 0.45,
        "A1": 0.45,
        "A2": 0.50,
        "B": 0.65,
        "D": 0.40,
    }

def should_exclude_by_days(expected_days: float) -> bool:
    return expected_days >= 5.5

def strategy_of_setup(setup: str) -> str:
    if setup.startswith("A"):
        return "PULLBACK"
    if setup.startswith("B"):
        return "BREAKOUT"
    if setup.startswith("D"):
        return "DISTORT"
    return "PULLBACK"

def strategy_caps(market_score: float, index_vol_regime: str) -> Dict[str, int]:
    ms = float(market_score)
    vol = str(index_vol_regime).upper()
    if ms >= 70:
        pull = 3
        brk = 2 if vol != "HIGH" else 1
        dist = 1
    elif ms >= 55:
        pull = 2
        brk = 1
        dist = 1
    else:
        pull = 1
        brk = 0
        dist = 1
    return {"PULLBACK": pull, "BREAKOUT": brk, "DISTORT": dist}

def pass_thresholds(c: Dict) -> Tuple[bool, str]:
    rr_min = rr_min_fixed()
    ev_min = ev_min_fixed()
    rot_min = rotation_min_by_setup()

    setup = str(c.get("setup", "A1"))
    rr = float(c.get("rr", 0.0))
    ev = float(c.get("exp_value", 0.0))
    rot = float(c.get("rotation_eff", 0.0))
    days = float(c.get("expected_days", 99.0))

    if should_exclude_by_days(days):
        return False, "想定日数>=6（除外）"
    if rr < rr_min:
        return False, f"RR<{rr_min}"
    if ev < ev_min:
        return False, f"期待値（補正）<{ev_min}"
    if rot < rot_min.get(setup, 0.50):
        return False, f"回転効率<{rot_min.get(setup, 0.50)}"
    return True, ""


def weekly_max_new() -> int:
    return 3

def max_display() -> int:
    return 5

def no_trade_conditions(mkt_score: int, delta3: float, macro_on: bool = False) -> bool:
    # MarketScoreはゲートにしないが、「新規を止める日」は運用上必要。
    # ここは撤退速度制御と同じ情報から"新規抑制"を判断する。
    ms = int(mkt_score)
    if macro_on:
        return False  # イベント警戒でも、指値小ロットで非常口は残す思想
    if ms <= 40:
        return True
    if delta3 <= -8.0:
        return True
    return False
