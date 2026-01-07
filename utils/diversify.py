# ============================================
# utils/diversify.py
# 分散・相関制御ロジック
# ============================================

from __future__ import annotations

from typing import List, Dict
import math


# --------------------------------------------
# 設定値
# --------------------------------------------
MAX_PER_SECTOR = 2          # 同一セクター最大数
MAX_CORR = 0.75             # 相関上限（超えたら除外）
MAX_RATE_SENSITIVE = 2      # 金利敏感系の最大数（保険・銀行など）


# --------------------------------------------
# セクター上限制御
# --------------------------------------------
def check_sector_limit(
    picks: List[Dict],
    candidate: Dict,
) -> bool:
    """
    すでに選ばれている銘柄とのセクター重複を制御
    """
    sector = candidate.get("sector")
    if sector is None:
        return True

    count = sum(1 for p in picks if p.get("sector") == sector)
    return count < MAX_PER_SECTOR


# --------------------------------------------
# 相関制御
# --------------------------------------------
def check_correlation(
    picks: List[Dict],
    candidate: Dict,
) -> bool:
    """
    相関係数が高すぎる場合は除外
    candidate["corr_map"] に
    {ticker: corr_value} が入っている前提
    """
    corr_map = candidate.get("corr_map", {})
    for p in picks:
        t = p.get("ticker")
        if t in corr_map and corr_map[t] >= MAX_CORR:
            return False
    return True


# --------------------------------------------
# マクロ感応度制御
# --------------------------------------------
def check_macro_exposure(
    picks: List[Dict],
    candidate: Dict,
) -> bool:
    """
    金利・景気感応度が偏りすぎないようにする
    """
    macro = candidate.get("macro", "other")

    if macro != "rate_sensitive":
        return True

    count = sum(1 for p in picks if p.get("macro") == "rate_sensitive")
    return count < MAX_RATE_SENSITIVE


# --------------------------------------------
# 総合チェック
# --------------------------------------------
def allow_add(
    picks: List[Dict],
    candidate: Dict,
) -> (bool, str):
    """
    分散観点で候補を追加してよいか判定
    """
    if not check_sector_limit(picks, candidate):
        return False, "セクター上限"

    if not check_correlation(picks, candidate):
        return False, "相関高"

    if not check_macro_exposure(picks, candidate):
        return False, "マクロ偏重"

    return True, ""