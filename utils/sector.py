from __future__ import annotations

"""
セクターは補助情報（選定理由にしない）
将来の観測用に残す（仕様）
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


UNIVERSE_PATH = "universe_jpx.csv"
MAX_TICKERS_PER_SECTOR = 15


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

    sec_col = "sector" if "sector" in df.columns else ("industry_big" if "industry_big" in df.columns else None)
    t_col = "ticker" if "ticker" in df.columns else ("code" if "code" in df.columns else None)
    if sec_col is None or t_col is None:
        return []

    out: List[Tuple[str, float]] = []
    for sec, sub in df.groupby(sec_col):
        tickers = sub[t_col].astype(str).tolist()[:MAX_TICKERS_PER_SECTOR]
        chgs = []
        for t in tickers:
            v = _fetch_change_5d(t)
            if np.isfinite(v):
                chgs.append(v)
        if chgs:
            out.append((str(sec), float(np.mean(chgs))))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]
