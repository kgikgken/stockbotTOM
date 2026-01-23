from __future__ import annotations

from typing import List, Dict


def apply_basic_diversify(cands: List[Dict], max_per_sector: int = 2) -> List[Dict]:
    out: List[Dict] = []
    counts = {}
    for c in cands:
        sec = c.get("sector") or "不明"
        counts.setdefault(sec, 0)
        if counts[sec] >= max_per_sector:
            continue
        out.append(c)
        counts[sec] += 1
    return out
