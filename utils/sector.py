from __future__ import annotations

import os
from typing import Dict, List, Tuple

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

def top_sectors_5d(top_n: int = 5, universe_path: str = UNIVERSE_PATH) -> List[Tuple[str, float]]:
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

def sector_rank_map(top_list: List[Tuple[str, float]]) -> Dict[str, int]:
    return {name: i + 1 for i, (name, _) in enumerate(top_list)}
