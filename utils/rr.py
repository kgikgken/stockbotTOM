from __future__ import annotations
import numpy as np
import pandas as pd

def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int, for_day: bool = False) -> dict:
    close = hist["Close"]
    price = close.iloc[-1]

    ma20 = close.rolling(20).mean().iloc[-1]
    entry = min(price, ma20)

    sl = entry * 0.97
    tp = entry * (1.08 if mkt_score >= 50 else 1.05)

    rr = (tp - entry) / (entry - sl)

    return {
        "entry": round(entry,1),
        "tp_pct": (tp/entry-1),
        "sl_pct": (sl/entry-1),
        "tp_price": round(tp,1),
        "sl_price": round(sl,1),
        "rr": rr
    }