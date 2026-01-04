from __future__ import annotations
import numpy as np
import pandas as pd

def correlation_filter(df: pd.DataFrame, corr: pd.DataFrame, cfg):
    picks = []
    sector_count = {}
    for _, r in df.sort_values("adj_ev", ascending=False).iterrows():
        sec = r["sector"]
        if sector_count.get(sec, 0) >= cfg.MAX_PER_SECTOR:
            continue
        ok = True
        for p in picks:
            if corr.loc[r["ticker"], p] > cfg.CORR_MAX:
                ok = False
        if ok:
            picks.append(r["ticker"])
            sector_count[sec] = sector_count.get(sec, 0) + 1
    return picks