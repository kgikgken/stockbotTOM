# ============================================
# utils/entry.py
# エントリー価格・GU判定・行動判定
# ============================================

from __future__ import annotations

from typing import Dict


# --------------------------------------------
# 設定
# --------------------------------------------
GU_THRESHOLD = 0.015   # 1.5%以上のGUは危険域


# --------------------------------------------
# GU判定
# --------------------------------------------
def is_gap_up(entry_price: float, current_price: float) -> bool:
    """
    エントリー価格に対して大きくGUしているか
    """
    if entry_price <= 0:
        return False
    return (current_price - entry_price) / entry_price >= GU_THRESHOLD


# --------------------------------------------
# 行動判定
# --------------------------------------------
def decide_action(
    entry_price: float,
    current_price: float,
    allow_trade: bool,
) -> str:
    """
    行動ラベルを返す
    """
    if not allow_trade:
        return "監視のみ"

    if is_gap_up(entry_price, current_price):
        return "寄り後再判定"

    if current_price <= entry_price:
        return "即IN可"

    return "指値待ち"


# --------------------------------------------
# エントリー帯生成
# --------------------------------------------
def entry_band(
    entry_price: float,
    atr: float,
    width_ratio: float = 0.25,
) -> (float, float):
    """
    押し目の許容帯を生成
    """
    if atr <= 0:
        return entry_price, entry_price

    w = atr * width_ratio
    return entry_price - w, entry_price + w


# --------------------------------------------
# 候補整形
# --------------------------------------------
def enrich_entry_fields(candidate: Dict, allow_trade: bool) -> Dict:
    """
    エントリー関連情報を付与
    """
    entry = candidate["entry"]
    current = candidate["current"]
    atr = candidate.get("atr", 0.0)

    band_low, band_high = entry_band(entry, atr)
    gu = is_gap_up(entry, current)
    action = decide_action(entry, current, allow_trade)

    candidate.update({
        "entry_low": round(band_low, 1),
        "entry_high": round(band_high, 1),
        "gu": gu,
        "action": action,
    })
    return candidate