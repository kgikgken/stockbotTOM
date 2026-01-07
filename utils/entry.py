# ============================================
# utils/entry.py
# エントリー帯と行動分類（追いかけ禁止）
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from utils.features import Features


@dataclass
class EntryPlan:
    entry: float
    band_low: float
    band_high: float
    action: str  # "即IN可" / "指値待ち" / "監視のみ"
    distance_atr: float


def calc_entry_plan(setup_type: str, f: Features) -> EntryPlan:
    """
    A: IN_center = SMA20、帯=±0.5ATR（A2は±0.6ATR）
    追いかけ禁止: Closeが帯上抜けしている場合は基本「指値待ち」 or 乖離が大きいなら「監視のみ」
    """
    atr = f.atr14 if f.atr14 > 0 else 0.0

    if setup_type == "A1":
        center = f.sma20
        w = 0.5 * atr
    elif setup_type == "A2":
        # A2は少し広く待てる帯
        center = f.sma20
        w = 0.6 * atr
    else:
        center = f.sma20
        w = 0.5 * atr

    band_low = center - w
    band_high = center + w

    # 乖離（ATR換算）
    dist = 0.0
    if atr > 0:
        dist = abs(f.close - center) / atr

    # GUは別で切る想定だが、ここでも保険で監視
    if f.gu_flag:
        return EntryPlan(entry=center, band_low=band_low, band_high=band_high, action="監視のみ", distance_atr=dist)

    # 行動分類
    if band_low <= f.close <= band_high:
        action = "即IN可"
    else:
        # 乖離大は監視
        if dist > 0.8:
            action = "監視のみ"
        else:
            action = "指値待ち"

    return EntryPlan(entry=center, band_low=band_low, band_high=band_high, action=action, distance_atr=dist)