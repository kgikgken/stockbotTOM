from __future__ import annotations

from typing import List, Dict
import numpy as np
import pandas as pd


def _corr_20d(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    if df_a is None or df_b is None:
        return 0.0
    if "ret1" not in df_a.columns or "ret1" not in df_b.columns:
        return 0.0

    a = df_a["ret1"].astype(float).dropna().tail(25)
    b = df_b["ret1"].astype(float).dropna().tail(25)
    if len(a) < 15 or len(b) < 15:
        return 0.0
    x = a.values[-min(len(a), len(b)):]
    y = b.values[-min(len(a), len(b)):]
    if len(x) < 15:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else 0.0


def pick_with_constraints(cands: List[Dict], max_final: int = 5, max_sector: int = 2, corr_limit: float = 0.75) -> List[Dict]:
    """
    - 同一セクター最大2
    - corr>0.75 は同時採用禁止
    """
    selected: List[Dict] = []
    sector_count = {}

    for c in cands:
        if len(selected) >= max_final:
            break

        sec = str(c.get("sector", "不明"))
        if sector_count.get(sec, 0) >= max_sector:
            c["reject_reason"] = "セクター上限"
            continue

        ok = True
        for s in selected:
            corr = _corr_20d(c.get("_df_ind"), s.get("_df_ind"))
            if corr > corr_limit:
                ok = False
                c["reject_reason"] = f"相関高({corr:.2f})"
                break

        if not ok:
            continue

        selected.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    return selected