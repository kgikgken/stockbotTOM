import numpy as np

def compute_rr_block(df, entry):
    atr = df["High"].rolling(14).max().iloc[-1] - df["Low"].rolling(14).min().iloc[-1]
    if atr <= 0:
        return None

    stop = entry - 1.2 * atr
    tp2 = entry + 3.0 * atr
    tp1 = entry + 1.5 * atr

    r = (tp2 - entry) / (entry - stop)
    return r, stop, tp1, tp2