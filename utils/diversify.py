from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict


def _corr20(t1: str, t2: str) -> float:
    try:
        d1 = yf.Ticker(t1).history(period="60d", auto_adjust=True)
        d2 = yf.Ticker(t2).history(period="60d", auto_adjust=True)
        if d1 is None or d2 is None or len(d1) < 25 or len(d2) < 25:
            return 0.0
        r1 = d1["Close"].astype(float).pct_change(fill_method=None).tail(20)
        r2 = d2["Close"].astype(float).pct_change(fill_method=None).tail(20)
        df = pd.concat([r1, r2], axis=1).dropna()
        if len(df) < 10:
            return 0.0
        c = float(df.corr().iloc[0, 1])
        if not np.isfinite(c):
            return 0.0
        return c
    except Exception:
        return 0.0


def apply_diversify(cands: List[Dict], sector_max: int = 2, corr_limit: float = 0.75) -> tuple[List[Dict], List[Dict]]:
    """
    - 同一セクター最大2
    - 相関 > 0.75 を同時採用しない
    """
    picked: List[Dict] = []
    watch: List[Dict] = []

    sec_count = {}

    for c in cands:
        sec = c.get("sector", "不明")
        if sec_count.get(sec, 0) >= sector_max:
            c["drop_reason"] = "セクター上限"
            watch.append(c)
            continue

        # 相関チェック（既採用と比較）
        too_corr = False
        for p in picked:
            corr = _corr20(c["ticker"], p["ticker"])
            if corr > corr_limit:
                c["drop_reason"] = f"相関高({corr:.2f})"
                too_corr = True
                break

        if too_corr:
            watch.append(c)
            continue

        picked.append(c)
        sec_count[sec] = sec_count.get(sec, 0) + 1

        if len(picked) >= 5:
            break

    return picked, watch