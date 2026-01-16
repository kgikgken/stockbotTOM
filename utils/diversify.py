from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd


def _corr(a: pd.Series, b: pd.Series) -> float:
    try:
        df = pd.concat([a, b], axis=1).dropna()
        if len(df) < 30:
            return 0.0
        return float(df.corr().iloc[0, 1])
    except Exception:
        return 0.0


def apply_diversification(
    cands: List[Dict],
    max_per_sector: int = 2,
    corr_limit: float = 0.75,
) -> List[Dict]:
    """Apply sector max and correlation constraint.

    Requires each candidate to include:
      - sector (string)
      - returns (pd.Series) daily returns index aligned
    """
    out: List[Dict] = []
    sector_counts: dict[str, int] = {}

    for c in cands:
        sec = str(c.get("sector", "不明"))
        if sector_counts.get(sec, 0) >= max_per_sector:
            continue

        ok = True
        for kept in out:
            r1 = c.get("returns")
            r2 = kept.get("returns")
            if isinstance(r1, pd.Series) and isinstance(r2, pd.Series):
                corr = _corr(r1, r2)
                if corr > corr_limit:
                    ok = False
                    break
        if not ok:
            continue

        out.append(c)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    return out
