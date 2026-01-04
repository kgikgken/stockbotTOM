# utils/diversify.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd


def corr_filter(candidates: List[Dict[str, Any]], corr_limit: float = 0.75) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    簡易：20日リターン相関が高いものを同時採用しない。
    candidates: 各要素に "ret_series" (pd.Series) がある想定
    """
    picked: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for c in candidates:
        ok = True
        for p in picked:
            s1 = c.get("ret_series")
            s2 = p.get("ret_series")
            if isinstance(s1, pd.Series) and isinstance(s2, pd.Series):
                df = pd.concat([s1, s2], axis=1).dropna()
                if len(df) >= 15:
                    corr = float(df.corr().iloc[0, 1])
                    if corr > corr_limit:
                        ok = False
                        c["reject_reason"] = f"相関高({corr:.2f})"
                        break
        if ok:
            picked.append(c)
        else:
            removed.append(c)

    return picked, removed