import numpy as np
import pandas as pd

def _atr(df, n=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]

def compute_tp_sl_rr(hist, mkt_score=50):
    close = hist["Close"]
    price = close.iloc[-1]
    atr = _atr(hist)

    entry = close.rolling(20).mean().iloc[-1] - 0.5 * atr
    entry = min(entry, price)

    sl = entry - 1.2 * atr
    tp = min(close.rolling(60).max().iloc[-1], entry + 3.0 * atr)

    rr = (tp - entry) / max(entry - sl, 1e-6)

    return {
        "entry": round(entry, 1),
        "sl_price": round(sl, 1),
        "tp_price": round(tp, 1),
        "rr": rr,
    }