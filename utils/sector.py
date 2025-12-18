from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf

UNIVERSE_PATH = "universe_jpx.csv"

def top_sectors_5d(top_n: int = 5):
    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    sec = "sector" if "sector" in df.columns else "industry_big"
    tcol = "ticker" if "ticker" in df.columns else "code"

    out = []
    for name, g in df.groupby(sec):
        chgs = []
        for t in g[tcol].head(20):
            try:
                d = yf.Ticker(t).history(period="6d", auto_adjust=True)
                if len(d) >= 2:
                    chgs.append((d["Close"].iloc[-1] / d["Close"].iloc[0] - 1) * 100)
            except Exception:
                pass
        if chgs:
            out.append((name, float(np.mean(chgs))))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]