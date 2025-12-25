import numpy as np
import pandas as pd

def judge_setup_and_pwin(hist: pd.DataFrame):
    close = hist["Close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    c = close.iloc[-1]
    m20 = ma20.iloc[-1]
    m50 = ma50.iloc[-1]

    rsi = 100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean()
                       / (close.diff().clip(upper=0).abs().rolling(14).mean() + 1e-9)))

    rsi_last = rsi.iloc[-1]

    if c > m20 > m50 and 35 <= rsi_last <= 55:
        return "A", 0.55
    if c > close.rolling(20).max().iloc[-2]:
        return "B", 0.48

    return "X", 0.0