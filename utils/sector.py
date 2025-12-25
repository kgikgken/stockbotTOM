import pandas as pd
import yfinance as yf
import numpy as np

def top_sectors_5d(n=5):
    df = pd.read_csv("universe_jpx.csv")
    out = []
    for s, g in df.groupby("sector"):
        chgs = []
        for t in g["ticker"][:20]:
            try:
                d = yf.Ticker(t).history(period="6d", auto_adjust=True)["Close"]
                chgs.append((d.iloc[-1]/d.iloc[0]-1)*100)
            except Exception:
                pass
        if chgs:
            out.append((s, np.mean(chgs)))
    return sorted(out, key=lambda x: x[1], reverse=True)[:n]