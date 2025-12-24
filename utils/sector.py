from __future__ import annotations

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "universe_jpx.csv"
MAX_TICKERS_PER_SECTOR = 20


def _fetch_change_5d(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="8d", auto_adjust=True)
        if df is None or df.empty or len(df) < 6:
            return np.nan
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[-6] - 1.0) * 100.0)
    except Exception:
        return np.nan


def top_sectors_5d(top_n: int = 5) -> List[Tuple[str, float]]:
    if not os.path.exists(UNIVERSE_PATH):
        return []
    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    sec_col = "sector" if "sector" in df.columns else ("industry_big" if "industry_big" in df.columns else None)
    t_col = "ticker" if "ticker" in df.columns else ("code" if "code" in df.columns else None)
    if not sec_col or not t_col:
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


def sector_rank_map(top_n: int = 5) -> Dict[str, int]:
    tops = top_sectors_5d(top_n=top_n)
    out: Dict[str, int] = {}
    for i, (s, _) in enumerate(tops, start=1):
        out[str(s)] = i
    return out
