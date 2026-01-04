from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Dict

MAX_TICKERS_PER_SECTOR = 20


def _fetch_change_5d(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return np.nan
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return np.nan


def top_sectors_5d(universe_path: str, top_n: int = 5) -> List[Tuple[str, float]]:
    if not os.path.exists(universe_path):
        return []
    try:
        df = pd.read_csv(universe_path)
    except Exception:
        return []

    sec_col = "sector" if "sector" in df.columns else ("industry_big" if "industry_big" in df.columns else None)
    t_col = "ticker" if "ticker" in df.columns else ("code" if "code" in df.columns else None)
    if sec_col is None or t_col is None:
        return []

    sectors: List[Tuple[str, float]] = []
    for sec_name, sub in df.groupby(sec_col):
        tickers = sub[t_col].astype(str).tolist()[:MAX_TICKERS_PER_SECTOR]
        chgs = []
        for t in tickers:
            chg = _fetch_change_5d(t)
            if np.isfinite(chg):
                chgs.append(chg)
        if chgs:
            sectors.append((str(sec_name), float(np.mean(chgs))))

    sectors.sort(key=lambda x: x[1], reverse=True)
    return sectors[:top_n]


def sector_rank_map(universe_path: str) -> Dict[str, int]:
    """
    セクターは“判断補助”として：ランキング（1=強い）を返す
    """
    tops = top_sectors_5d(universe_path=universe_path, top_n=50)
    rank: Dict[str, int] = {}
    for i, (s, _) in enumerate(tops, 1):
        rank[s] = i
    return rank