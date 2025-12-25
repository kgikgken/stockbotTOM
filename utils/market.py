import numpy as np
import yfinance as yf

def enhance_market_score():
    def chg(sym):
        df = yf.Ticker(sym).history(period="6d", auto_adjust=True)
        return (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100 if len(df) >= 2 else 0

    base = 50 + (chg("^TOPX") + chg("^N225")) / 2
    score = int(np.clip(base, 0, 100))

    comment = "中立" if 45 <= score <= 60 else "強め" if score > 60 else "弱め"
    return {"score": score, "comment": comment}