from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from utils.features import FeaturePack

@dataclass
class SetupResult:
    setup_type: Optional[str]  # "A" | "B" | None
    reasons: List[str]

def decide_setup(feat: FeaturePack) -> SetupResult:
    reasons: List[str] = []
    c = feat.close

    # Setup A: トレンド押し目
    a_ok = True
    if not (np.isfinite(c) and np.isfinite(feat.sma20) and np.isfinite(feat.sma50)):
        a_ok = False
    else:
        if not (c > feat.sma20 > feat.sma50):
            a_ok = False
        if not (feat.sma20_slope_5d > 0):
            a_ok = False
        if not (abs(c - feat.sma20) <= 0.8 * feat.atr14):
            a_ok = False
        if not (40 <= feat.rsi14 <= 62):
            a_ok = False

    # Setup B: ブレイク
    b_ok = True
    if not (np.isfinite(c) and np.isfinite(feat.hh20) and np.isfinite(feat.vol) and np.isfinite(feat.vol_ma20)):
        b_ok = False
    else:
        if not (c > feat.hh20):
            b_ok = False
        if not (feat.vol >= 1.5 * feat.vol_ma20):
            b_ok = False

    # 優先はA（押し目）
    if a_ok:
        return SetupResult(setup_type="A", reasons=reasons)

    if b_ok:
        return SetupResult(setup_type="B", reasons=reasons)

    return SetupResult(setup_type=None, reasons=["setup条件外"])
