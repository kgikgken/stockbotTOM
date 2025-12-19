from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple

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
    df = pd.read_csv(UNIVERSE_PATH)
    sec_col = "sector" if "sector" in df.columns else "industry_big"
    t_col = "ticker" if "ticker" in df.columns else "code"

    sectors = []
    for sec, g in df.groupby(sec_col):
        chgs = [_fetch_change_5d(t) for t in g[t_col][:MAX_TICKERS_PER_SECTOR]]
        chgs = [c for c in chgs if np.isfinite(c)]
        if chgs:
            sectors.append((sec, float(np.mean(chgs))))

    sectors.sort(key=lambda x: x[1], reverse=True)
    return sectors[:top_n]