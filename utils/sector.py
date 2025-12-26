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
        df = yf.Ticker(ticker).history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return np.nan
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return np.nan

def top_sectors_5d(top_n: int = 5) -> List[Tuple[str, float]]:
    if not os.path.exists(UNIVERSE_PATH):
        return []
    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    if "sector" in df.columns:
        sec_col = "sector"
    elif "industry_big" in df.columns:
        sec_col = "industry_big"
    else:
        return []

    if "ticker" in df.columns:
        t_col = "ticker"
    elif "code" in df.columns:
        t_col = "code"
    else:
        return []

    sectors: List[Tuple[str, float]] = []
    for sec_name, sub in df.groupby(sec_col):
        tickers = sub[t_col].astype(str).tolist()
        if not tickers:
            continue
        tickers = tickers[:MAX_TICKERS_PER_SECTOR]

        chgs = []
        for t in tickers:
            chg = _fetch_change_5d(t)
            if np.isfinite(chg):
                chgs.append(chg)
        if chgs:
            sectors.append((str(sec_name), float(np.mean(chgs))))

    sectors.sort(key=lambda x: x[1], reverse=True)
    return sectors[:top_n]

def get_sector_rank_map(top_n: int = 5) -> Tuple[List[Tuple[str, float]], Dict[str, int], List[str]]:
    tops = top_sectors_5d(top_n=top_n)
    rank_map: Dict[str, int] = {}
    top_names: List[str] = []
    for i, (name, _) in enumerate(tops):
        rank_map[str(name)] = i + 1
        top_names.append(str(name))
    return tops, rank_map, top_names
