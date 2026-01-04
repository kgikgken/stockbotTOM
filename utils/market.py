from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd, yfinance as yf
from utils.util import clamp

@dataclass
class MarketState:
    index_ticker: str
    market_score: float
    momentum_3d: float
    close: float
    sma10: float
    sma20: float
    sma50: float

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def get_market_state(index_ticker: str, sma_fast: int, sma_slow: int, sma_risk: int) -> MarketState:
    df = yf.download(index_ticker, period="240d", progress=False)
    close = df["Close"].astype(float)

    sma10 = _sma(close, sma_risk).iloc[-1]
    sma20 = _sma(close, sma_fast).iloc[-1]
    sma50 = _sma(close, sma_slow).iloc[-1]

    score = 50.0
    last = close.iloc[-1]
    score += 15 if last > sma50 else -15
    score += 15 if sma20 > sma50 else -10
    score += 10 if last > sma10 else -10
    score = clamp(score, 0, 100)

    # momentum
    score_3d = score  # simplified
    mom3 = score - score_3d

    return MarketState(index_ticker, score, mom3, last, sma10, sma20, sma50)