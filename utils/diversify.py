from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def apply_sector_cap(candidates: List[Dict], cap: int = 2) -> List[Dict]:
    out: List[Dict] = []
    seen: dict[str, int] = {}
    for cand in candidates:
        sector = str(cand.get("sector", "Other"))
        if seen.get(sector, 0) >= int(cap):
            continue
        out.append(cand)
        seen[sector] = seen.get(sector, 0) + 1
    return out


def _corr(df1: pd.DataFrame, df2: pd.DataFrame, lookback: int = 60) -> float:
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return float("nan")
    r1 = df1["Close"].astype(float).pct_change().tail(lookback)
    r2 = df2["Close"].astype(float).pct_change().tail(lookback)
    joined = pd.concat([r1, r2], axis=1).dropna()
    if len(joined) < max(20, lookback // 2):
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def apply_corr_filter(candidates: List[Dict], ohlc_map: Dict[str, pd.DataFrame], threshold: float = 0.87) -> List[Dict]:
    accepted: List[Dict] = []
    for cand in candidates:
        keep = True
        df = ohlc_map.get(str(cand.get("ticker")))
        for taken in accepted:
            df_taken = ohlc_map.get(str(taken.get("ticker")))
            corr = _corr(df, df_taken)
            if np.isfinite(corr) and corr >= threshold and str(cand.get("sector")) == str(taken.get("sector")):
                keep = False
                break
        if keep:
            accepted.append(cand)
    return accepted
