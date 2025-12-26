# utils/market.py
from __future__ import annotations
import yfinance as yf
import numpy as np

def _chg(symbol: str) -> float:
    df = yf.Ticker(symbol).history(period="6d", auto_adjust=True)
    return (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100 if len(df) >= 2 else 0

def enhance_market_score() -> dict:
    nk = _chg("^N225")
    tp = _chg("^TOPX")
    score = 50 + (nk + tp) / 2
    score = int(np.clip(score, 0, 100))
    return {
        "score": score,
        "delta3d": int((nk + tp) / 2),
    }