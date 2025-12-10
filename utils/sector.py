from __future__ import annotations

import os
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "universe_jpx.csv"
MAX_TICKERS_PER_SECTOR = 20


def _fetch_change_5d(ticker: str) -> float:
    """
    単純な 5日騰落率（%）
    """
    try:
        df = yf.Ticker(ticker).history(period="6d")
        if df is None or df.empty or len(df) < 5:
            return np.nan
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return np.nan


def top_sectors_5d() -> List[Dict[str, float]]:
    """
    universe_jpx.csv からセクターごとの代表銘柄を取り、
    5日騰落率の平均を計算して上位順に返す。

    戻り値: [{"sector": str, "chg": float}, ...]
    """
    if not os.path.exists(UNIVERSE_PATH):
        return []

    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    if "sector" not in df.columns:
        return []

    ticker_col = "ticker" if "ticker" in df.columns else "code"

    sectors: List[Dict[str, float]] = []

    for sec_name, sub in df.groupby("sector"):
        tickers = sub[ticker_col].astype(str).unique().tolist()
        if not tickers:
            continue

        # 多すぎると重いので制限
        tickers = tickers[:MAX_TICKERS_PER_SECTOR]

        chgs = []
        for t in tickers:
            chg = _fetch_change_5d(t)
            if np.isfinite(chg):
                chgs.append(chg)

        if chgs:
            avg_chg = float(np.mean(chgs))
            sectors.append({"sector": sec_name, "chg": avg_chg})

    sectors.sort(key=lambda x: x["chg"], reverse=True)
    return sectors