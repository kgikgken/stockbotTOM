from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _corr20(a: pd.Series, b: pd.Series) -> float:
    try:
        x = a.astype(float).pct_change().dropna().tail(20)
        y = b.astype(float).pct_change().dropna().tail(20)
        n = min(len(x), len(y))
        if n < 10:
            return 0.0
        c = float(np.corrcoef(x[-n:], y[-n:])[0, 1])
        if not np.isfinite(c):
            return 0.0
        return c
    except Exception:
        return 0.0


def diversify_candidates(
    candidates: List[Dict],
    max_per_sector: int = 2,
    corr_threshold: float = 0.75,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Keep list with sector cap and high-corr removal.
    Returns: (kept, moved_to_watch)
    candidates must include:
      - sector
      - adj_ev
      - hist_close (pd.Series) for corr
    """
    kept: List[Dict] = []
    watch: List[Dict] = []

    sector_count: Dict[str, int] = {}

    for c in candidates:
        sec = str(c.get("sector", "不明"))
        if sector_count.get(sec, 0) >= max_per_sector:
            c2 = dict(c)
            c2["watch_reason"] = "セクター上限"
            watch.append(c2)
            continue

        # correlation check against kept
        too_corr = False
        worst_pair = None
        for k in kept:
            corr = _corr20(c["hist_close"], k["hist_close"])
            if corr > corr_threshold:
                too_corr = True
                worst_pair = (k, corr)
                break

        if too_corr and worst_pair is not None:
            k, corr = worst_pair
            # keep higher adjEV
            if c["adj_ev"] > k["adj_ev"]:
                # move existing kept to watch
                kept.remove(k)
                k2 = dict(k)
                k2["watch_reason"] = f"相関高({corr:.2f})"
                watch.append(k2)

                kept.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            else:
                c2 = dict(c)
                c2["watch_reason"] = f"相関高({corr:.2f})"
                watch.append(c2)
            continue

        kept.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    return kept, watch