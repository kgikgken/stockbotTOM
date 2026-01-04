from __future__ import annotations

import os
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import yfinance as yf


def _fetch_change_5d(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return float("nan")
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return float("nan")


def build_sector_ranking(universe_path: str, top_n: int = 5, max_tickers_per_sector: int = 25) -> List[Tuple[str, float]]:
    """
    セクターは “判断補助” 用。選定理由ではない。
    """
    if not universe_path or (not os.path.exists(universe_path)):
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

    sectors: List[Tuple[str, float]] = []

    for sec_name, sub in df.groupby(sec_col):
        tickers = sub[t_col].astype(str).tolist()
        tickers = [t.strip() for t in tickers if t and str(t).strip()]
        if not tickers:
            continue
        tickers = tickers[:max_tickers_per_sector]

        chgs = []
        for t in tickers:
            v = _fetch_change_5d(t)
            if np.isfinite(v):
                chgs.append(v)
        if chgs:
            sectors.append((str(sec_name), float(np.mean(chgs))))

    sectors.sort(key=lambda x: x[1], reverse=True)
    return sectors[:top_n]