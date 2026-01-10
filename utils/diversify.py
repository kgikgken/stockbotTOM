from __future__ import annotations

from typing import List, Dict
import numpy as np
import pandas as pd

def _corr(a: pd.Series, b: pd.Series) -> float:
    try:
        if a is None or b is None:
            return 0.0
        df = pd.concat([a, b], axis=1).dropna()
        if len(df) < 40:
            return 0.0
        return float(df.iloc[:,0].corr(df.iloc[:,1]))
    except Exception:
        return 0.0

def apply_diversification(candidates: List[Dict], max_per_sector: int = 2, corr_max: float = 0.75) -> List[Dict]:
    """Greedy select by score_key, enforcing sector count and correlation limit."""
    selected: List[Dict] = []
    sector_count: Dict[str, int] = {}

    for c in candidates:
        sec = str(c.get("sector", "不明"))
        if sector_count.get(sec, 0) >= max_per_sector:
            continue

        ok = True
        for s in selected:
            corr = _corr(c.get("_ret60"), s.get("_ret60"))
            if np.isfinite(corr) and corr > corr_max:
                ok = False
                break
        if not ok:
            continue

        selected.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    # cleanup temp keys
    for c in selected:
        if "_ret60" in c:
            c.pop("_ret60", None)
    return selected
