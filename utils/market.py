import yfinance as yf
import numpy as np

def _chg(sym):
    d = yf.Ticker(sym).history(period="6d", auto_adjust=True)["Close"]
    return (d.iloc[-1]/d.iloc[0]-1)*100

def enhance_market_score():
    base = 50 + (_chg("^TOPX")+_chg("^N225"))/2
    score = int(np.clip(base, 0, 100))
    comment = "中立" if score >= 50 else "弱め"
    return {"score": score, "comment": comment}

def market_delta_3d():
    try:
        d = yf.Ticker("^TOPX").history(period="6d", auto_adjust=True)["Close"]
        return int((d.iloc[-1]/d.iloc[-4]-1)*100)
    except Exception:
        return 0