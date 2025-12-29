from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "universe_jpx.csv"
MAX_TICKERS_PER_SECTOR = 25


def _fetch_change_5d(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return float("nan")
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return float("nan")


def top_sectors_5d(universe_path: str = UNIVERSE_PATH, top_n: int = 5) -> List[Tuple[str, float]]:
    if not universe_path or not os.path.exists(universe_path):
        return []

    try:
        df = pd.read_csv(universe_path)
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

    out: List[Tuple[str, float]] = []
    for sec_name, sub in df.groupby(sec_col):
        tickers = sub[t_col].astype(str).tolist()
        if not tickers:
            continue
        tickers = tickers[:MAX_TICKERS_PER_SECTOR]

        chgs = []
        for t in tickers:
            v = _fetch_change_5d(t)
            if np.isfinite(v):
                chgs.append(v)
        if chgs:
            out.append((str(sec_name), float(np.mean(chgs))))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]


def sector_rank_map(universe_path: str = UNIVERSE_PATH, top_n: int = 33) -> Dict[str, int]:
    """1が最強。データ不足は 999。"""
    if not universe_path or not os.path.exists(universe_path):
        return {}

    try:
        df = pd.read_csv(universe_path)
    except Exception:
        return {}

    if "sector" in df.columns:
        sec_col = "sector"
    elif "industry_big" in df.columns:
        sec_col = "industry_big"
    else:
        return {}

    if "ticker" in df.columns:
        t_col = "ticker"
    elif "code" in df.columns:
        t_col = "code"
    else:
        return {}

    vals: List[Tuple[str, float]] = []
    for sec_name, sub in df.groupby(sec_col):
        tickers = sub[t_col].astype(str).tolist()[:MAX_TICKERS_PER_SECTOR]
        chgs = []
        for t in tickers:
            v = _fetch_change_5d(t)
            if np.isfinite(v):
                chgs.append(v)
        if chgs:
            vals.append((str(sec_name), float(np.mean(chgs))))

    vals.sort(key=lambda x: x[1], reverse=True)
    mp: Dict[str, int] = {}
    for i, (sec, _) in enumerate(vals[:top_n], start=1):
        mp[sec] = i
    return mp