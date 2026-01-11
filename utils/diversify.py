from __future__ import annotations

from typing import List, Dict
import numpy as np
import pandas as pd

def _corr(a: pd.Series, b: pd.Series) -> float:
    try:
        df = pd.concat([a, b], axis=1).dropna()
        if len(df) < 40:
            return 0.0
        return float(df.iloc[:,0].pct_change().corr(df.iloc[:,1].pct_change()))
    except Exception:
        return 0.0

def diversify(cands: List[Dict], *, sector_cap: int = 2, corr_max: float = 0.75) -> List[Dict]:
    out: List[Dict] = []
    sec_cnt = {}

    for c in cands:
        sec = str(c.get("sector", "不明"))
        if sec_cnt.get(sec, 0) >= sector_cap:
            continue

        ok = True
        for chosen in out:
            s1 = c.get("_close_series")
            s2 = chosen.get("_close_series")
            if isinstance(s1, pd.Series) and isinstance(s2, pd.Series):
                corr = _corr(s1, s2)
                if np.isfinite(corr) and corr > corr_max:
                    ok = False
                    break
        if not ok:
            continue

        out.append(c)
        sec_cnt[sec] = sec_cnt.get(sec, 0) + 1

    for c in out:
        c.pop("_close_series", None)
    return out
