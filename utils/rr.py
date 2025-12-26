import numpy as np
import pandas as pd

def compute_trade_plan(df: pd.DataFrame, setup: str, mkt_score: int) -> dict:
    close = df["Close"]
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]

    if setup == "A":
        entry = close.rolling(20).mean().iloc[-1]
        stop = entry - 1.2 * atr
    else:
        entry = close.rolling(20).max().iloc[-2]
        stop = entry - 1.0 * atr

    r = entry - stop
    tp1 = entry + 1.5 * r
    tp2 = entry + 3.0 * r

    R = (tp2 - entry) / r if r > 0 else 0.0

    return dict(
        entry=entry,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        R=R,
        in_low=entry - 0.5 * atr,
        in_high=entry + 0.5 * atr,
    )