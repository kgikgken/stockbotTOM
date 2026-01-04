from __future__ import annotations

import numpy as np

from utils.util import clamp


def entry_zone(setup: str, ind: dict) -> dict:
    """
    IN帯（追いかけ禁止を機械化）
    A：IN_center=MA20、帯=±0.5ATR
    B：IN_center=HH20、帯=±0.3ATR
    """
    c = float(ind["close"])
    atr = float(ind["atr"])
    ma20 = float(ind["ma20"])
    hh20 = float(ind["hh20"])

    if not (np.isfinite(c) and np.isfinite(atr) and atr > 0):
        return {"in_center": np.nan, "in_low": np.nan, "in_high": np.nan, "deviation": np.nan, "action": "監視のみ"}

    if setup == "A":
        center = ma20
        band = 0.5 * atr
    else:
        center = hh20
        band = 0.3 * atr

    if not np.isfinite(center):
        return {"in_center": np.nan, "in_low": np.nan, "in_high": np.nan, "deviation": np.nan, "action": "監視のみ"}

    in_low = center - band
    in_high = center + band

    # 乖離率（distance/ATR）
    dev = abs(c - center) / atr if atr > 0 else 999.0

    # 行動分類
    # EXEC_NOW：帯の中 & dev小
    # LIMIT_WAIT：帯外だが dev<=0.8
    # WATCH_ONLY：dev>0.8
    if in_low <= c <= in_high and dev <= 0.6:
        action = "即IN可"
    elif dev <= 0.8:
        action = "指値待ち"
    else:
        action = "監視のみ"

    return {
        "in_center": float(center),
        "in_low": float(in_low),
        "in_high": float(in_high),
        "deviation": float(dev),
        "action": action,
    }