from __future__ import annotations
import numpy as np
import pandas as pd

def score_daytrade_candidate(hist_d: pd.DataFrame, mkt_score: int = 50) -> float:
    if len(hist_d) < 60:
        return 0.0
    close = hist_d["Close"]
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()

    if close.iloc[-1] > ma5.iloc[-1] > ma20.iloc[-1]:
        return 80 + (mkt_score - 50) * 0.3
    return 0.0