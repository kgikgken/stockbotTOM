from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.features import FeaturePack


@dataclass
class SetupResult:
    setup_type: str  # "A" or "B" or "-"
    ok: bool
    reason: str


def detect_setup(fp: FeaturePack, hh20_break: bool, vol_break_ok: bool) -> SetupResult:
    """
    Setup A: トレンド押し目（最優先）
    Setup B: ブレイク（厳選）
    """
    # A
    if fp.trend_ok:
        # 押し目：abs(Close−MA20) <= 0.8ATR
        if np.isfinite(fp.pullback_dist_atr) and fp.pullback_dist_atr <= 0.8:
            # RSI過熱なし
            if np.isfinite(fp.rsi14) and 40 <= fp.rsi14 <= 62:
                return SetupResult("A", True, "MA構造+押し目+RSI適正")
            return SetupResult("-", False, "RSI条件外")
        return SetupResult("-", False, "押し目距離>0.8ATR")
    # B
    if hh20_break and vol_break_ok:
        return SetupResult("B", True, "HH20ブレイク+出来高増")
    return SetupResult("-", False, "Setup不成立")