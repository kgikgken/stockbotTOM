import yfinance as yf
import numpy as np

def _chg(symbol):
    df = yf.Ticker(symbol).history(period="6d", auto_adjust=True)
    return (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

def enhance_market_score():
    nk = _chg("^N225")
    tp = _chg("^TOPX")
    score = int(np.clip(50 + (nk + tp) / 2, 0, 100))
    return {"score": score}

def calc_market_delta3d():
    try:
        df = yf.Ticker("^TOPX").history(period="5d", auto_adjust=True)
        s1 = df["Close"].iloc[-1]
        s3 = df["Close"].iloc[-4]
        return int((s1 / s3 - 1) * 100)
    except Exception:
        return 0