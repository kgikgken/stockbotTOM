from __future__ import annotations

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import read_csv_safely, pick_ticker_column, pick_sector_column


UNIVERSE_PATH = "universe_jpx.csv"
MAX_TICKERS_PER_SECTOR = 25


def _chg_5d(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return float("nan")
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return float("nan")


def top_sectors_5d(top_n: int = 5) -> List[Tuple[str, float]]:
    df = read_csv_safely(UNIVERSE_PATH)
    if df.empty:
        return []

    t_col = pick_ticker_column(df)
    s_col = pick_sector_column(df)
    if not t_col or not s_col:
        return []

    out: List[Tuple[str, float]] = []
    for sec, sub in df.groupby(s_col):
        tickers = sub[t_col].astype(str).tolist()
        if not tickers:
            continue
        tickers = tickers[:MAX_TICKERS_PER_SECTOR]
        vals = []
        for t in tickers:
            v = _chg_5d(t)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            out.append((str(sec), float(np.mean(vals))))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]


def sector_rank_map() -> Dict[str, int]:
    """
    セクターは「判断補助」。選定理由ではない。
    ただし Pwin(代理) の加点で使う。
    """
    tops = top_sectors_5d(top_n=33)
    mp: Dict[str, int] = {}
    for i, (name, _) in enumerate(tops, start=1):
        mp[name] = i
    return mp