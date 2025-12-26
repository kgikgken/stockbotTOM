# utils/rr.py
from __future__ import annotations
import pandas as pd

def calc_rr_block(hist: pd.DataFrame, zone: dict, mkt_score: int) -> dict:
    atr = zone["atr"]
    entry = zone["center"]
    stop = entry - 1.2 * atr
    tp2 = entry + 3.0 * (entry - stop)
    tp1 = entry + 1.5 * (entry - stop)

    rr = (tp2 - entry) / (entry - stop)
    expected_days = (tp2 - entry) / atr

    if mkt_score < 50:
        tp2 *= 0.95

    return {
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "expected_days": expected_days,
    }