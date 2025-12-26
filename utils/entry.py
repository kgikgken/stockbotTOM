from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from utils.features import FeaturePack

@dataclass
class EntryPlan:
    in_center: float
    in_low: float
    in_high: float
    gu_flag: bool
    dist_atr: float
    action: str  # "EXEC_NOW" | "LIMIT_WAIT" | "WATCH_ONLY"
    action_jp: str

def build_entry_plan(feat: FeaturePack, setup_type: str) -> EntryPlan:
    atr = feat.atr14 if np.isfinite(feat.atr14) and feat.atr14 > 0 else max(feat.close * 0.01, 1.0)

    if setup_type == "A":
        center = float(feat.sma20)
        band = 0.5 * atr
    else:  # "B"
        center = float(feat.hh20)
        band = 0.3 * atr

    in_low = center - band
    in_high = center + band

    # GU判定（仕様通り）
    gu = bool(np.isfinite(feat.open) and np.isfinite(feat.prev_close) and (feat.open > feat.prev_close + 1.0 * atr))

    # 乖離率
    dist = float(abs(feat.close - center) / atr) if atr > 0 else 999.0

    # Action
    if gu or dist > 0.8:
        action = "WATCH_ONLY"
        jp = "監視のみ"
    else:
        # 帯の中なら即IN、外なら指値
        if in_low <= feat.close <= in_high and dist <= 0.3:
            action = "EXEC_NOW"
            jp = "即IN可"
        else:
            action = "LIMIT_WAIT"
            jp = "指値待ち"

    return EntryPlan(
        in_center=float(center),
        in_low=float(in_low),
        in_high=float(in_high),
        gu_flag=bool(gu),
        dist_atr=float(dist),
        action=str(action),
        action_jp=str(jp),
    )
