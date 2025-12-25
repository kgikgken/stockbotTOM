import pandas as pd
import yfinance as yf
import numpy as np

UNIVERSE_PATH = "universe_jpx.csv"

def top_sectors_5d(n=5):
    df = pd.read_csv(UNIVERSE_PATH)
    t_col = "ticker" if "ticker" in df.columns else "code"
    s_col = "sector" if "sector" in df.columns else "industry_big"

    out = []
    for sec, g in df.groupby(s_col):
        chgs = []
        for t in g[t_col].head(10):
            try:
                h = yf.Ticker(t).history(period="6d", auto_adjust=True)
                chgs.append((h["Close"].iloc[-1] / h["Close"].iloc[0] - 1) * 100)
            except:
                pass
        if chgs:
            out.append((sec, np.mean(chgs)))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:n]