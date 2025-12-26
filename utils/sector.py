# utils/sector.py
from __future__ import annotations
import pandas as pd
import yfinance as yf
import numpy as np

UNIVERSE_PATH = "universe_jpx.csv"

def top_sectors_5d(top_n: int = 5):
    df = pd.read_csv(UNIVERSE_PATH)
    out = []
    for sec, sub in df.groupby("sector"):
        chgs = []
        for t in sub["ticker"].head(10):
            h = yf.Ticker(t).history(period="6d", auto_adjust=True)
            if len(h) >= 2:
                chgs.append((h["Close"].iloc[-1] / h["Close"].iloc[0] - 1) * 100)
        if chgs:
            out.append((sec, float(np.mean(chgs))))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]