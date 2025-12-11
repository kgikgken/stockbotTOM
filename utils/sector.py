from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "universe_jpx.csv"
MAX_TICKERS_PER_SECTOR = 20


def _change_5d(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="6d")
        if df is None or df.empty or len(df) < 2:
            return np.nan
        close = df["Close"].astype(float)
        return float(close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
    except Exception:
        return np.nan


def top_sectors_5d() -> List[Tuple[str, float]]:
    """
    universe_jpx.csv から各セクターの代表銘柄を取り 5日騰落率の平均を返す
    戻り値: [(sector, pct_change), ...]
    """
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

    sectors: List[Tuple[str, float]] = []

    for sec_name, sub in df.groupby(sec_col):
        if "ticker" in sub.columns:
            tickers = sub["ticker"].astype(str).tolist()
        elif "code" in sub.columns:
            tickers = sub["code"].astype(str).tolist()
        else:
            continue

        if not tickers:
            continue

        tickers = tickers[:MAX_TICKERS_PER_SECTOR]

        chgs = []
        for t in tickers:
            ch = _change_5d(t)
            if np.isfinite(ch):
                chgs.append(ch)

        if chgs:
            avg_chg = float(np.mean(chgs))
            sectors.append((sec_name, avg_chg))

    sectors.sort(key=lambda x: x[1], reverse=True)
    return sectors