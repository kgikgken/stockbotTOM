from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_entry_zone(setup_type: str, feat: dict) -> Dict:
    """
    IN_center + band
    A: center=MA20, band=±0.5ATR
    B: center=HH20, band=±0.3ATR
    GU_flag = Open > PrevClose + 1.0ATR
    distance_atr = abs(Close-center)/ATR
    Action:
      EXEC_NOW: within band & not GU & distance<=0.8
      LIMIT_WAIT: outside band but distance<=0.8 & not GU
      WATCH_ONLY: GU or distance>0.8
    """
    c = float(feat["close"])
    atr = float(feat["atr"])
    o = float(feat["open"])
    prev = float(feat["prev_close"])

    if not np.isfinite(atr) or atr <= 0:
        atr = max(c * 0.01, 1.0)

    if setup_type == "A":
        center = float(feat["ma20"])
        band = 0.5 * atr
    else:
        center = float(feat["hh20"])
        band = 0.3 * atr

    in_low = center - band
    in_high = center + band

    gu_flag = bool(o > (prev + 1.0 * atr))

    dist_atr = abs(c - center) / atr if atr > 0 else 999.0

    if gu_flag or dist_atr > 0.8:
        action = "監視のみ"
    else:
        if in_low <= c <= in_high:
            action = "即IN可"
        else:
            action = "指値待ち"

    return {
        "center": float(round(center, 1)),
        "in_low": float(round(in_low, 1)),
        "in_high": float(round(in_high, 1)),
        "gu_flag": gu_flag,
        "dist_atr": float(dist_atr),
        "action": action,
    }