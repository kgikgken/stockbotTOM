import numpy as np
import pandas as pd

def trend_gate(hist: pd.DataFrame) -> bool:
    close = hist["Close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    if close.iloc[-1] <= ma20.iloc[-1] <= ma50.iloc[-1]:
        return False

    slope = ma20.diff().iloc[-1]
    if slope <= 0:
        return False

    high60 = close.rolling(60).max().iloc[-1]
    off = (close.iloc[-1] / high60 - 1) * 100
    return off >= -20

def score_stock(hist: pd.DataFrame) -> float:
    close = hist["Close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    rsi = 100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean()
                      / (-close.diff().clip(upper=0).rolling(14).mean() + 1e-9)))

    score = 0
    if close.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]:
        score += 40
    if rsi.iloc[-1] < 50:
        score += 30
    if hist["Volume"].iloc[-1] * close.iloc[-1] > 1e8:
        score += 30

    return float(min(score, 100))

def strong_in_only(hist: pd.DataFrame):
    close = hist["Close"]
    ma20 = close.rolling(20).mean()
    rsi = 100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean()
                      / (-close.diff().clip(upper=0).rolling(14).mean() + 1e-9)))

    if 30 <= rsi.iloc[-1] <= 50 and close.iloc[-1] >= ma20.iloc[-1] * 0.97:
        entry = float(ma20.iloc[-1])
        return "強IN", entry
    return "NG", 0.0