from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np
import yfinance as yf

from utils.screen_logic import Candidate


def _corr(tickers: List[str], lookback: int = 60) -> pd.DataFrame:
    if len(tickers) <= 1:
        return pd.DataFrame()
    end = pd.Timestamp.today() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback*2)
    try:
        data = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        closes = data["Close"].dropna(how="all")
        rets = closes.pct_change().dropna()
        if rets.empty:
            return pd.DataFrame()
        return rets.corr()
    except Exception:
        return pd.DataFrame()


def diversify(cands: List[Candidate], max_per_sector: int = 2, corr_limit: float = 0.75, max_total: int = 5) -> List[Candidate]:
    # sector cap first
    picked: List[Candidate] = []
    sector_counts: dict[str, int] = {}

    # Precompute correlation on top candidates only (speed)
    top = cands[: min(30, len(cands))]
    corr = _corr([c.ticker for c in top])

    def ok_corr(new: Candidate) -> bool:
        if corr.empty or not picked:
            return True
        for p in picked:
            try:
                v = float(corr.loc[new.ticker, p.ticker])
                if np.isnan(v):
                    continue
                if v >= corr_limit:
                    return False
            except Exception:
                continue
        return True

    for c in cands:
        if len(picked) >= max_total:
            break
        sec = c.sector
        if sector_counts.get(sec, 0) >= max_per_sector:
            continue
        if not ok_corr(c):
            continue
        picked.append(c)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    return picked
