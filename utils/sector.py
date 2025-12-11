from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "universe_jpx.csv"

# セクターごとに何銘柄まで見るか（重すぎ防止）
MAX_TICKERS_PER_SECTOR = 20


def _fetch_change_5d(ticker: str) -> float:
    """5日騰落率（％）"""
    try:
        df = yf.Ticker(ticker).history(period="6d")
        if df is None or len(df) < 5:
            return np.nan
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return np.nan


def top_sectors_5d() -> List[Tuple[str, float]]:
    """
    universe_jpx.csv を見て、各セクターの代表銘柄を拾い、
    5日騰落率の平均を算出 → 上位順に返す。
    戻り値例: [("銀行業", 3.4), ("電気機器", 1.9), ...]
    """
    if not os.path.exists(UNIVERSE_PATH):
        return []

    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    # sector or industry_big
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

    for name, sub in df.groupby(sec_col):
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
            avg_chg = float(np.mean(chgs))
            sectors.append((name, avg_chg))

    sectors.sort(key=lambda x: x[1], reverse=True)
    return sectors