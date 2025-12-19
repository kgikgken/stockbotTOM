from __future__ import annotations
import numpy as np
import pandas as pd


def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> dict:
    close = hist["Close"].astype(float)
    price = float(close.iloc[-1])

    ma20 = close.rolling(20).mean().iloc[-1]
    atr = close.pct_change().rolling(14).std().iloc[-1] * price
    atr = max(atr, price * 0.01)

    entry = min(ma20, price * 0.995)
    sl_price = entry - atr
    tp_price = entry + atr * 3.0

    sl_pct = (sl_price / entry - 1.0)
    tp_pct = (tp_price / entry - 1.0)
    rr = tp_pct / abs(sl_pct)

    return {
        "entry": round(entry, 1),
        "tp_price": round(tp_price, 1),
        "sl_price": round(sl_price, 1),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "rr": rr,
        "entry_basis": "pullback",
    }