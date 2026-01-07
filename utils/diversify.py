# ============================================
# utils/diversify.py
# 分散・集中リスク制御
# - 同一セクター制限
# - 相関制限
# - 週次新規制限補助
# ============================================

from __future__ import annotations

from typing import List, Dict
import numpy as np


# --------------------------------------------
# 同一セクター制限
# --------------------------------------------
def filter_by_sector_limit(
    candidates: List[dict],
    per_sector_limit: int,
) -> List[dict]:
    """
    同一セクターの採用数を制限
    """
    result = []
    sector_count: Dict[str, int] = {}

    for c in candidates:
        sector = c.get("sector", "UNKNOWN")
        cnt = sector_count.get(sector, 0)

        if cnt >= per_sector_limit:
            c["reject_reason"] = "セクター上限"
            continue

        sector_count[sector] = cnt + 1
        result.append(c)

    return result


# --------------------------------------------
# 相関制限
# --------------------------------------------
def filter_by_correlation(
    candidates: List[dict],
    corr_block_threshold: float,
) -> List[dict]:
    """
    20日リターン相関が高い銘柄を同時採用しない
    ※ candidates は AdjEV 降順で来る前提
    """
    accepted: List[dict] = []

    for c in candidates:
        corr_vec = c.get("corr_with", {})  # {ticker: corr}
        blocked = False

        for a in accepted:
            t = a.get("ticker")
            if t in corr_vec and abs(corr_vec[t]) >= corr_block_threshold:
                c["reject_reason"] = f"相関高({corr_vec[t]:.2f})"
                blocked = True
                break

        if not blocked:
            accepted.append(c)

    return accepted


# --------------------------------------------
# 最終分散フィルタ
# --------------------------------------------
def apply_diversification(
    candidates: List[dict],
    per_sector_limit: int,
    corr_block_threshold: float,
) -> List[dict]:
    """
    分散ルールを順に適用
    """
    step1 = filter_by_sector_limit(candidates, per_sector_limit)
    step2 = filter_by_correlation(step1, corr_block_threshold)
    return step2