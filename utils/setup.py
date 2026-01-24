from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def liquidity_filters(row: Dict, adv_yen_min: float = 300_000_000) -> bool:
    adv = row.get("adv20", float("nan"))
    try:
        return float(adv) >= adv_yen_min
    except Exception:
        return False


def build_setup_info(row: Dict) -> Optional[str]:
    pull = float(row.get("pullback_score", 0.0))
    mom = float(row.get("momentum_score", 0.0))

    if pull >= 80 and mom >= 55:
        return "A1-Strong"
    if pull >= 60:
        return "A1"
    return None


def rday_min_by_setup(setup: str) -> float:
    if setup == "A1-Strong":
        return 0.50
    if setup == "A1":
        return 0.45
    return 0.65
