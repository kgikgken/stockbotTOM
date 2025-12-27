from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HIGH_BETA_SECTORS = {"電気機器", "情報・通信業", "精密機器", "輸送用機器"}

def _returns_20(hist: pd.DataFrame) -> pd.Series | None:
    try:
        close = hist["Close"].astype(float)
        if len(close) < 25:
            return None
        r = close.pct_change(fill_method=None).dropna()
        return r.tail(20)
    except Exception:
        return None

def diversify_select(
    candidates: List[Dict],
    histories: Dict[str, pd.DataFrame],
    max_final: int = 5,
    max_per_sector: int = 2,
    max_highbeta_sector: int = 1,
    corr_threshold: float = 0.75,
) -> Tuple[List[Dict], List[Dict]]:
    selected: List[Dict] = []
    dropped: List[Dict] = []

    rets: Dict[str, pd.Series] = {}
    for c in candidates:
        t = c["ticker"]
        h = histories.get(t)
        s = _returns_20(h) if h is not None else None
        if s is not None and len(s) >= 10:
            rets[t] = s

    sector_count: Dict[str, int] = {}

    for c in candidates:
        if len(selected) >= max_final:
            break

        sec = str(c.get("sector", "不明"))
        limit = max_highbeta_sector if sec in HIGH_BETA_SECTORS else max_per_sector
        if sector_count.get(sec, 0) >= limit:
            d = dict(c); d["drop_reason"] = "セクター上限"
            dropped.append(d)
            continue

        ok = True
        for s in selected:
            t1, t2 = c["ticker"], s["ticker"]
            r1, r2 = rets.get(t1), rets.get(t2)
            if r1 is None or r2 is None:
                continue
            n = min(len(r1), len(r2))
            if n < 10:
                continue
            corr = float(pd.concat([r1.tail(n).reset_index(drop=True), r2.tail(n).reset_index(drop=True)], axis=1).corr().iloc[0,1])
            if np.isfinite(corr) and corr > corr_threshold:
                ok = False
                break

        if not ok:
            d = dict(c); d["drop_reason"] = "相関高"
            dropped.append(d)
            continue

        selected.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    return selected, dropped
