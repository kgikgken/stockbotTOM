# ============================================
# utils/entry.py
# エントリー判定ロジック
# - 追いかけ禁止
# - INゾーン判定
# - GU（ギャップアップ）制御
# - 行動分類（EXEC / WAIT / WATCH）
# ============================================

from __future__ import annotations

from typing import Dict
from utils.util import clamp


# --------------------------------------------
# ギャップアップ判定
# --------------------------------------------
def is_gap_up(
    open_price: float,
    prev_close: float,
    atr: float,
    atr_threshold: float = 1.0,
) -> bool:
    """
    前日終値 + ATR×threshold を超えて寄ったら GU
    """
    if atr <= 0:
        return False
    return open_price > prev_close + atr * atr_threshold


# --------------------------------------------
# INゾーン計算
# --------------------------------------------
def calc_entry_zone(
    setup_type: str,
    sma20: float,
    break_line: float | None,
    atr: float,
) -> Dict[str, float]:
    """
    Setup別 INゾーン（帯）を返す
    """
    if setup_type.startswith("A"):
        center = sma20
        width = atr * 0.5
    else:  # B（ブレイク）
        center = break_line if break_line is not None else sma20
        width = atr * 0.3

    return {
        "center": center,
        "low": center - width,
        "high": center + width,
    }


# --------------------------------------------
# 乖離率
# --------------------------------------------
def calc_entry_deviation(
    price: float,
    center: float,
    atr: float,
) -> float:
    if atr <= 0:
        return 999.0
    return abs(price - center) / atr


# --------------------------------------------
# 行動判定
# --------------------------------------------
def judge_action(
    price: float,
    entry_zone: Dict[str, float],
    deviation: float,
    gu_flag: bool,
    deviation_limit: float = 0.8,
) -> str:
    """
    EXEC_NOW / LIMIT_WAIT / WATCH_ONLY
    """
    if gu_flag:
        return "WATCH_ONLY"

    if deviation > deviation_limit:
        return "WATCH_ONLY"

    if entry_zone["low"] <= price <= entry_zone["high"]:
        return "EXEC_NOW"

    return "LIMIT_WAIT"


# --------------------------------------------
# エントリー総合判定
# --------------------------------------------
def evaluate_entry(
    setup_type: str,
    price: float,
    open_price: float,
    prev_close: float,
    sma20: float,
    break_line: float | None,
    atr: float,
) -> Dict[str, object]:
    """
    エントリー関連を一括評価
    """
    gu = is_gap_up(open_price, prev_close, atr)
    zone = calc_entry_zone(setup_type, sma20, break_line, atr)
    deviation = calc_entry_deviation(price, zone["center"], atr)
    action = judge_action(price, zone, deviation, gu)

    return {
        "gu_flag": gu,
        "entry_center": zone["center"],
        "entry_low": zone["low"],
        "entry_high": zone["high"],
        "entry_deviation": deviation,
        "action": action,
    }