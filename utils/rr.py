import numpy as np
import pandas as pd

def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int):
    close = hist["Close"]
    price = close.iloc[-1]

    atr = (hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1]
    if not np.isfinite(atr):
        atr = price * 0.02

    entry = close.rolling(20).mean().iloc[-1]
    sl = entry - atr
    tp = min(entry + atr * 3.5, close.rolling(60).max().iloc[-1])

    rr = (tp - entry) / (entry - sl)
    return {
        "entry": float(entry),
        "tp_price": float(tp),
        "sl_price": float(sl),
        "rr": float(rr),
    }