from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def corr20(a_close: pd.Series, b_close: pd.Series) -> float:
    try:
        if a_close is None or b_close is None:
            return 0.0
        if len(a_close) < 25 or len(b_close) < 25:
            return 0.0
        a = a_close.pct_change(fill_method=None).tail(20)
        b = b_close.pct_change(fill_method=None).tail(20)
        df = pd.concat([a, b], axis=1).dropna()
        if len(df) < 10:
            return 0.0
        c = float(df.corr().iloc[0, 1])
        if not np.isfinite(c):
            return 0.0
        return c
    except Exception:
        return 0.0


def apply_diversify(cands: List[Dict], sector_cap: int = 2, corr_cap: float = 0.75) -> Tuple[List[Dict], List[Dict]]:
    """
    同一セクター最大2、相関>0.75 同時採用禁止
    戻り: (selected, dropped_with_reason)
    """
    selected: List[Dict] = []
    dropped: List[Dict] = []
    sector_counts: Dict[str, int] = {}

    for c in cands:
        sec = str(c.get("sector", "不明"))
        if sector_counts.get(sec, 0) >= sector_cap:
            c2 = dict(c)
            c2["drop_reason"] = "セクター上限"
            dropped.append(c2)
            continue

        # correlation check
        too_corr = False
        for s in selected:
            ca = c.get("_close_series")
            cb = s.get("_close_series")
            if ca is None or cb is None:
                continue
            co = corr20(ca, cb)
            if co > corr_cap:
                too_corr = True
                break

        if too_corr:
            c2 = dict(c)
            c2["drop_reason"] = f"相関高({corr_cap:.2f}超)"
            dropped.append(c2)
            continue

        selected.append(c)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    return selected, dropped