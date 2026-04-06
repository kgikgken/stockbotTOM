from __future__ import annotations

from typing import Dict

from utils.util import safe_float


WIN_RATE_BY_SETUP = {
    "A1-Strong": 0.48,
    "A1": 0.44,
    "A2": 0.38,
    "B": 0.42,
}


def calc_ev(plan: Dict) -> float:
    rr = safe_float(plan.get("rr"), 0.0)
    setup = str(plan.get("setup", ""))
    win = WIN_RATE_BY_SETUP.get(setup, 0.40)
    return float(win * rr - (1.0 - win))


def pass_thresholds(plan: Dict) -> bool:
    rr = safe_float(plan.get("rr"), 0.0)
    ev = safe_float(plan.get("ev"), calc_ev(plan))
    return rr >= 1.8 and ev > -0.05
