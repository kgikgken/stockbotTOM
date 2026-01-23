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
    """Classify a candidate into one of the supported setups.

    v2.3+ spec:
    - Main strategy: Pullback trend follow (A1 / A1-Strong)
    - Secondary: Initial breakout (B)
    - Exception: Supply-demand distortion (D)

    The screener must remain mechanical; thresholds are fixed here.
    """

    pull = float(row.get("pullback_score", 0.0))
    mom = float(row.get("momentum_score", 0.0))
    brk = float(row.get("breakout_score", 0.0))
    dist = float(row.get("distortion_score", 0.0))

    # Prefer the primary strategy when it is high quality.
    if pull >= 85 and mom >= 55:
        return "A1-Strong"
    if pull >= 70:
        return "A1"

    # Secondary / exception strategies.
    if brk >= 75:
        return "B"
    if dist >= 80:
        return "D"
    return None


def rday_min_by_setup(setup: str) -> float:
    if setup == "A1-Strong":
        return 0.50
    if setup == "A1":
        return 0.45
    if setup == "B":
        return 0.65
    if setup == "D":
        # Distortion trades are fast but size is constrained; keep a modest bar.
        return 0.50
    return 0.65


def reach_prob_base(setup: str) -> float:
    """Base reach probability by setup.

    This replaces explicit Pwin modelling. It is intentionally coarse
    and is modulated by per-setup quality scores in screen_logic.
    """
    if setup == "A1-Strong":
        return 0.62
    if setup == "A1":
        return 0.55
    if setup == "B":
        return 0.45
    if setup == "D":
        return 0.35
    return 0.50
