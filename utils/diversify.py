from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class DiversifyConfig:
    max_per_sector: int = 2
    corr_lookback: int = 60
    corr_threshold: float = 0.75


def _returns(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    return c.pct_change(fill_method=None).dropna()


def _corr(a: pd.DataFrame, b: pd.DataFrame, lookback: int) -> float:
    try:
        ra = _returns(a).tail(lookback)
        rb = _returns(b).tail(lookback)
        joined = pd.concat([ra, rb], axis=1).dropna()
        if joined.shape[0] < max(20, lookback // 2):
            return 0.0
        return float(joined.corr().iloc[0, 1])
    except Exception:
        return 0.0


def apply_diversification(
    candidates: List[Dict[str, object]],
    histories: Dict[str, pd.DataFrame],
    cfg: DiversifyConfig,
) -> List[Dict[str, object]]:
    """
    仕様：同一セクター最大2、相関>0.75同時採用禁止
    """
    picked: List[Dict[str, object]] = []
    sector_count: Dict[str, int] = {}

    for c in candidates:
        t = str(c.get("ticker"))
        sec = str(c.get("sector", "不明"))

        if sector_count.get(sec, 0) >= cfg.max_per_sector:
            continue

        ok = True
        for p in picked:
            t2 = str(p.get("ticker"))
            a = histories.get(t)
            b = histories.get(t2)
            if a is None or b is None:
                continue
            corr = _corr(a, b, cfg.corr_lookback)
            if corr >= cfg.corr_threshold:
                ok = False
                break

        if not ok:
            continue

        picked.append(c)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    return picked
