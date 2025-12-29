from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DiversifyDecision:
    ok: bool
    reason: str


def _corr(a: pd.Series, b: pd.Series) -> float:
    try:
        x = a.astype(float).pct_change().dropna().tail(20)
        y = b.astype(float).pct_change().dropna().tail(20)
        if len(x) < 10 or len(y) < 10:
            return float("nan")
        m = pd.concat([x, y], axis=1).dropna()
        if len(m) < 10:
            return float("nan")
        return float(m.corr().iloc[0, 1])
    except Exception:
        return float("nan")


def select_with_constraints(
    candidates: List[dict],
    max_final: int,
    max_same_sector: int,
    corr_max: float,
) -> Tuple[List[dict], List[dict]]:
    """AdjEV降順で、セクター上限/相関上限を守って採用。落ちたものは watch に理由を付ける。"""
    picked: List[dict] = []
    watch: List[dict] = []

    sector_cnt: Dict[str, int] = {}
    closes: Dict[str, pd.Series] = {}  # ticker -> close series（20d相関用）

    for c in candidates:
        t = c["ticker"]
        sec = c.get("sector", "不明")
        sector_cnt.setdefault(sec, 0)

        # セクター上限
        if sector_cnt[sec] >= max_same_sector:
            c2 = dict(c)
            c2["reason"] = "セクター上限"
            watch.append(c2)
            continue

        # 相関上限
        corr_ng = False
        worst = 0.0
        worst_t = ""
        if c.get("_close_series") is not None:
            closes[t] = c["_close_series"]
        for p in picked:
            t2 = p["ticker"]
            s1 = closes.get(t)
            s2 = closes.get(t2)
            if s1 is None or s2 is None:
                continue
            cc = _corr(s1, s2)
            if np.isfinite(cc) and cc > corr_max:
                corr_ng = True
                worst = cc
                worst_t = t2
                break

        if corr_ng:
            c2 = dict(c)
            c2["reason"] = f"相関高({worst:.2f})"
            c2["corr_with"] = worst_t
            watch.append(c2)
            continue

        picked.append(c)
        sector_cnt[sec] += 1
        if len(picked) >= max_final:
            break

    # 余った候補を監視へ（理由付きで）
    for c in candidates[len(picked):]:
        if len(watch) >= 50:
            break
        if c.get("reason"):
            watch.append(c)
        else:
            c2 = dict(c)
            c2["reason"] = "採用枠外"
            watch.append(c2)

    return picked, watch