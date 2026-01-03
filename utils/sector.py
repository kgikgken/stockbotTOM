from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.util import fetch_history, safe_float


UNIVERSE_PATH_DEFAULT = "universe_jpx.csv"
MAX_TICKERS_PER_SECTOR = 25


def _chg_5d(ticker: str) -> float:
    df = fetch_history(ticker, period="10d")
    if df is None or df.empty or len(df) < 6:
        return float("nan")
    c = df["Close"].astype(float)
    return float((c.iloc[-1] / c.iloc[-6] - 1.0) * 100.0)


def compute_sector_5d_rank(universe_path: str, top_n: int = 5) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
    """
    Returns:
      - top list: [(sector, avg_5d%), ...]
      - rank map: {sector: rank(1..)}
    """
    if not universe_path:
        universe_path = UNIVERSE_PATH_DEFAULT
    if not os.path.exists(universe_path):
        return [], {}

    try:
        df = pd.read_csv(universe_path)
    except Exception:
        return [], {}

    if "sector" in df.columns:
        sec_col = "sector"
    elif "industry_big" in df.columns:
        sec_col = "industry_big"
    else:
        sec_col = None

    if "ticker" in df.columns:
        t_col = "ticker"
    elif "code" in df.columns:
        t_col = "code"
    else:
        return [], {}

    if sec_col is None:
        return [], {}

    rows: List[Tuple[str, float]] = []
    for sec_name, sub in df.groupby(sec_col):
        tickers = sub[t_col].astype(str).tolist()
        tickers = [t.strip() for t in tickers if t and str(t).strip()]
        if not tickers:
            continue
        tickers = tickers[:MAX_TICKERS_PER_SECTOR]

        chgs = []
        for t in tickers:
            v = _chg_5d(t)
            if np.isfinite(v):
                chgs.append(v)
        if chgs:
            rows.append((str(sec_name), float(np.mean(chgs))))

    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:top_n]

    rank_map: Dict[str, int] = {}
    for i, (sec, _) in enumerate(rows, start=1):
        rank_map[sec] = i

    return top, rank_map