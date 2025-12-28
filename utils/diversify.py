from __future__ import annotations

from typing import List, Dict


def apply_diversification(candidates: List[Dict]):
    final = []
    dropped = []
    sector_count = {}

    for c in sorted(candidates, key=lambda x: x["adj_ev"], reverse=True):
        sec = c["sector"]
        sector_count.setdefault(sec, 0)

        if sector_count[sec] >= 2:
            dropped.append({**c, "reason": "セクター上限"})
            continue

        sector_count[sec] += 1
        final.append(c)

        if len(final) >= 5:
            break

    return final, dropped