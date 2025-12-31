from __future__ import annotations

from typing import List, Tuple

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


def top_sectors_5d(universe_path: str, top_n: int = 5, max_tickers_per_sector: int = 20) -> List[Tuple[str, float]]:
    try:
        uni = pd.read_csv(universe_path)
    except Exception:
        return []

    if "sector" not in uni.columns and "industry_big" in uni.columns:
        uni["sector"] = uni["industry_big"]
    if "sector" not in uni.columns:
        return []

    if "ticker" not in uni.columns and "code" in uni.columns:
        uni = uni.rename(columns={"code": "ticker"})
    if "ticker" not in uni.columns:
        return []

    out: List[Tuple[str, float]] = []
    for sec, sub in uni.groupby("sector"):
        tickers = sub["ticker"].astype(str).tolist()
        if not tickers:
            continue
        tickers = tickers[:max_tickers_per_sector]

        chgs = []
        for t in tickers:
            v = _fetch_change_5d(t)
            if np.isfinite(v):
                chgs.append(float(v))
        if chgs:
            out.append((str(sec), float(np.mean(chgs))))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]