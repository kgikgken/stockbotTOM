from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class DiversifyConfig:
    max_per_sector: int = 2
    corr_lookback: int = 20
    corr_limit: float = 0.75


def _corr(a: pd.Series, b: pd.Series) -> float:
    try:
        x = a.pct_change(fill_method=None).dropna()
        y = b.pct_change(fill_method=None).dropna()
        n = min(len(x), len(y))
        if n < 10:
            return 0.0
        return float(np.corrcoef(x.tail(n), y.tail(n))[0, 1])
    except Exception:
        return 0.0


def apply_diversify(
    candidates: List[Dict],
    price_hist_map: Dict[str, pd.DataFrame],
    cfg: DiversifyConfig = DiversifyConfig(),
) -> Tuple[List[Dict], List[Dict]]:
    """
    - 同一セクター最大2
    - 20日相関 > 0.75 は同時採用しない
    戻り: (selected, rejected_with_reason)
    """
    selected: List[Dict] = []
    rejected: List[Dict] = []

    sector_count: Dict[str, int] = {}

    for c in candidates:
        sec = str(c.get("sector", "不明"))
        ticker = str(c.get("ticker", ""))

        # セクター上限
        if sector_count.get(sec, 0) >= cfg.max_per_sector:
            rc = dict(c)
            rc["reject_reason"] = "セクター上限"
            rejected.append(rc)
            continue

        # 相関チェック
        ok = True
        for s in selected:
            t2 = str(s.get("ticker", ""))
            h1 = price_hist_map.get(ticker)
            h2 = price_hist_map.get(t2)
            if h1 is None or h2 is None:
                continue
            if "Close" not in h1.columns or "Close" not in h2.columns:
                continue
            corr = _corr(h1["Close"].astype(float).tail(cfg.corr_lookback),
                         h2["Close"].astype(float).tail(cfg.corr_lookback))
            if corr >= cfg.corr_limit:
                ok = False
                rc = dict(c)
                rc["reject_reason"] = f"相関高({corr:.2f})"
                rejected.append(rc)
                break

        if not ok:
            continue

        selected.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    return selected, rejected