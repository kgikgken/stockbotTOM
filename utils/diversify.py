from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _ret_series(hist: pd.DataFrame, n: int = 20) -> pd.Series:
    close = hist["Close"].astype(float)
    ret = close.pct_change(fill_method=None)
    return ret.tail(n).reset_index(drop=True)


def corr(a: pd.Series, b: pd.Series) -> float:
    try:
        if len(a) != len(b) or len(a) < 5:
            return 0.0
        v = float(np.corrcoef(a.values, b.values)[0, 1])
        if not np.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def apply_constraints(
    candidates: List[Dict],
    max_final: int = 5,
    sector_max: int = 2,
    macro_max: int = 2,
    corr_limit: float = 0.75,
) -> Tuple[List[Dict], List[Dict]]:
    """
    candidates: already sorted by priority (AdjEV, R/day...)
    Enforce:
      - same sector <= sector_max
      - same macro_tag <= macro_max
      - 20d return corr <= corr_limit
    Return: (selected, watchlist_rejected_with_reason)
    """
    selected: List[Dict] = []
    rejected: List[Dict] = []

    sec_cnt: Dict[str, int] = {}
    macro_cnt: Dict[str, int] = {}

    for c in candidates:
        if len(selected) >= max_final:
            c2 = dict(c)
            c2["reject_reason"] = "枠上限"
            rejected.append(c2)
            continue

        sec = str(c.get("sector", "不明"))
        macro = str(c.get("macro_tag", "other"))

        if sec_cnt.get(sec, 0) >= sector_max:
            c2 = dict(c)
            c2["reject_reason"] = "セクター上限"
            rejected.append(c2)
            continue

        if macro_cnt.get(macro, 0) >= macro_max:
            c2 = dict(c)
            c2["reject_reason"] = "マクロ上限"
            rejected.append(c2)
            continue

        # corr check
        ok = True
        for s in selected:
            a = c.get("_ret20")
            b = s.get("_ret20")
            if a is not None and b is not None:
                r = corr(a, b)
                if r > corr_limit:
                    ok = False
                    c2 = dict(c)
                    c2["reject_reason"] = f"相関高({r:.2f})"
                    rejected.append(c2)
                    break
        if not ok:
            continue

        selected.append(c)
        sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
        macro_cnt[macro] = macro_cnt.get(macro, 0) + 1

    return selected, rejected