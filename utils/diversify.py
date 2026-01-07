# ============================================
# utils/diversify.py
# 分散制約（同一セクター上限、相関制約）
# ============================================

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def corr(a: pd.Series, b: pd.Series) -> float:
    try:
        x = a.dropna()
        y = b.dropna()
        n = min(len(x), len(y))
        if n < 20:
            return 0.0
        x = x.iloc[-n:]
        y = y.iloc[-n:]
        return float(np.corrcoef(x.values, y.values)[0, 1])
    except Exception:
        return 0.0


def pick_with_constraints(
    candidates: List[dict],
    max_final: int = 5,
    max_per_sector: int = 2,
    corr_threshold: float = 0.75,
) -> Tuple[List[dict], List[dict]]:
    """
    candidates: dictに
      - sector
      - close_series（相関用, pd.Series）
    を持っている想定
    """
    selected: List[dict] = []
    watch: List[dict] = []

    sector_count: Dict[str, int] = {}

    for c in candidates:
        if len(selected) >= max_final:
            watch.append({**c, "drop_reason": "枠上限"})
            continue

        sec = str(c.get("sector") or "不明")
        if sector_count.get(sec, 0) >= max_per_sector:
            watch.append({**c, "drop_reason": "セクター上限"})
            continue

        # 相関
        too_corr = False
        for s in selected:
            a = c.get("close_series")
            b = s.get("close_series")
            if isinstance(a, pd.Series) and isinstance(b, pd.Series):
                r = corr(a, b)
                if r > corr_threshold:
                    too_corr = True
                    break

        if too_corr:
            watch.append({**c, "drop_reason": f"相関高({corr_threshold}超)"})
            continue

        selected.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    return selected, watch