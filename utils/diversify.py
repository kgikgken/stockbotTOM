from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

@dataclass
class DiversifyResult:
    picks: List[dict]
    watch_added: List[dict]

def _corr_20d(a: pd.Series, b: pd.Series) -> float:
    try:
        aa = a.pct_change(fill_method=None).dropna().tail(21)
        bb = b.pct_change(fill_method=None).dropna().tail(21)
        df = pd.concat([aa.rename("a"), bb.rename("b")], axis=1).dropna()
        if len(df) < 15:
            return 0.0
        return float(df["a"].corr(df["b"]))
    except Exception:
        return 0.0

def apply_diversify(candidates: List[dict], max_picks: int = 5, sector_max: int = 2, corr_max: float = 0.75) -> DiversifyResult:
    picks: List[dict] = []
    watch: List[dict] = []
    sector_count: Dict[str, int] = {}

    for c in candidates:
        if len(picks) >= max_picks:
            break

        sec = str(c.get("sector","不明"))
        if sector_count.get(sec, 0) >= sector_max:
            c2 = dict(c)
            c2["watch_reason"] = "セクター上限"
            watch.append(c2)
            continue

        ok = True
        for p in picks:
            try:
                corr = _corr_20d(c["hist_close"], p["hist_close"])
            except Exception:
                corr = 0.0
            if corr > corr_max:
                ok = False
                c2 = dict(c)
                c2["watch_reason"] = f"相関高い(corr>{corr_max:.2f})"
                watch.append(c2)
                break

        if not ok:
            continue

        picks.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    return DiversifyResult(picks=picks, watch_added=watch)
