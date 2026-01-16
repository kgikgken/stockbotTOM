from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.util import corr_60d

def apply_sector_cap(cands: List[Dict], max_per_sector: int = 2) -> List[Dict]:
    out = []
    cnt = {}
    for c in cands:
        sec = str(c.get("sector", "不明"))
        cnt.setdefault(sec, 0)
        if cnt[sec] >= max_per_sector:
            continue
        out.append(c)
        cnt[sec] += 1
    return out

def apply_corr_filter(cands: List[Dict], ohlc_map: Dict[str, object], max_corr: float = 0.75) -> List[Dict]:
    out: List[Dict] = []
    for c in cands:
        t = c.get("ticker")
        ok = True
        for chosen in out:
            t2 = chosen.get("ticker")
            df1 = ohlc_map.get(t)
            df2 = ohlc_map.get(t2)
            if df1 is None or df2 is None:
                continue
            cr = corr_60d(df1, df2)
            if np.isfinite(cr) and cr > max_corr:
                ok = False
                break
        if ok:
            out.append(c)
    return out
