from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf

MAX_TICKERS_PER_SECTOR = 20
UNIVERSE_PATH = "universe_jpx.csv"


def _fetch_change_5d(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="6d", auto_adjust=True)
        if df is None or len(df) < 5:
            return np.nan
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return np.nan


def top_sectors_5d() -> list[tuple[str, float]]:
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

    # ticker列名吸収
    if "ticker" in df.columns:
        t_col = "ticker"
    elif "code" in df.columns:
        t_col = "code"
    else:
        return []

    sectors: list[tuple[str, float]] = []
    for name, sub in df.groupby(sec_col):
        tickers = sub[t_col].astype(str).tolist()
        tickers = [t.replace(".T.T", ".T") for t in tickers if t]
        tickers = tickers[:MAX_TICKERS_PER_SECTOR]
        if not tickers:
            continue

        chgs = []
        for t in tickers:
            chg = _fetch_change_5d(t)
            if np.isfinite(chg):
                chgs.append(chg)

        if chgs:
            sectors.append((str(name), float(np.mean(chgs))))

    sectors.sort(key=lambda x: x[1], reverse=True)
    return sectors