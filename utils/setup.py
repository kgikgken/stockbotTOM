from __future__ import annotations

from typing import Tuple

import numpy as np

from utils.util import clamp


def detect_setup(feat: dict) -> Tuple[str, str]:
    """
    Return: (setup_type, reason_if_fail)
    setup_type: "A" | "B" | "-"
    """
    c = feat["close"]
    ma20 = feat["ma20"]
    ma50 = feat["ma50"]
    atr = feat["atr"]
    rsi = feat["rsi"]
    ma20_slope5 = feat["ma20_slope5"]

    if not (np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50) and np.isfinite(atr) and np.isfinite(rsi)):
        return "-", "データ不足"

    # Setup A: trend pullback
    cond_trend = (c > ma20 > ma50) and (ma20_slope5 > 0)
    cond_pullback = abs(c - ma20) <= 0.8 * atr
    cond_rsi = (40 <= rsi <= 62)

    if cond_trend and cond_pullback and cond_rsi:
        return "A", ""

    # Setup B: breakout
    hh20 = feat["hh20"]
    vol_last = feat["vol_last"]
    vol_ma20 = feat["vol_ma20"]

    cond_break = np.isfinite(hh20) and (c > hh20)
    cond_vol = np.isfinite(vol_last) and np.isfinite(vol_ma20) and (vol_ma20 > 0) and (vol_last >= 1.5 * vol_ma20)

    if cond_break and cond_vol and (c > ma50):
        return "B", ""

    return "-", "形(A/B)不一致"