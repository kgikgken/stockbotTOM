from dataclasses import dataclass
from typing import Tuple

import numpy as np

from utils.features import Feat


@dataclass
class EntryPlan:
    in_center: float
    in_low: float
    in_high: float
    deviation_atr: float     # distance(Close, center)/ATR
    gu_flag: bool
    action: str              # EXEC_NOW / LIMIT_WAIT / WATCH_ONLY


def _safe(v: float) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def make_entry_plan(feat: Feat, setup_type: str) -> EntryPlan:
    atr = feat.atr14
    c = feat.close
    pc = feat.prev_close
    o = feat.open_ if np.isfinite(feat.open_) else c

    if not np.isfinite(atr) or atr <= 0:
        atr = max(c * 0.01, 1.0)

    if setup_type == "A":
        center = feat.ma20
        band = 0.5 * atr
    elif setup_type == "B":
        center = feat.hh20
        band = 0.3 * atr
    else:
        center = c
        band = 0.5 * atr

    in_low = center - band
    in_high = center + band

    # GU判定：Open > PrevClose + 1ATR（Openが無いケースは Close で近似）
    gu_flag = bool(np.isfinite(o) and np.isfinite(pc) and o > pc + 1.0 * atr)
    if not np.isfinite(o) and np.isfinite(c) and np.isfinite(pc):
        gu_flag = bool(c > pc + 1.0 * atr)

    deviation = abs(c - center) / atr if np.isfinite(c) else 999.0

    # 行動分類（仕様書）
    if gu_flag:
        action = "WATCH_ONLY"
    else:
        if in_low <= c <= in_high and deviation <= 0.8:
            action = "EXEC_NOW"
        elif deviation <= 0.8:
            action = "LIMIT_WAIT"
        else:
            action = "WATCH_ONLY"

    return EntryPlan(
        in_center=_safe(center),
        in_low=_safe(in_low),
        in_high=_safe(in_high),
        deviation_atr=float(deviation),
        gu_flag=gu_flag,
        action=action,
    )