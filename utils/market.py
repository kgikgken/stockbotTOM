from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import ema, rsi14, pct_change


@dataclass
class MarketSnapshot:
    score: int
    score_label: str
    delta_3d: float
    futures_pct: float
    futures_ticker: str
    risk_on: bool


def _label(score: int) -> str:
    if score >= 75:
        return "強い"
    if score >= 55:
        return "普通"
    if score >= 45:
        return "中立"
    return "弱い"


def fetch_futures_pct(ticker: str = "NKD=F") -> float:
    try:
        df = yf.download(ticker, period="7d", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty or len(df) < 2:
            return float("nan")
        c0 = float(df["Close"].iloc[-2])
        c1 = float(df["Close"].iloc[-1])
        return pct_change(c0, c1) * 100.0
    except Exception:
        return float("nan")


def compute_market_score() -> MarketSnapshot:
    futures_ticker = "NKD=F"
    futures_pct = fetch_futures_pct(futures_ticker)

    try:
        df = yf.download("1321.T", period="6mo", interval="1d", auto_adjust=False, progress=False)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty or len(df) < 60:
        score = 50
        return MarketSnapshot(
            score=score,
            score_label=_label(score),
            delta_3d=0.0,
            futures_pct=float(futures_pct),
            futures_ticker=futures_ticker,
            risk_on=(np.isfinite(futures_pct) and futures_pct >= 1.5),
        )

    close = df["Close"]
    e25 = ema(close, 25)
    e50 = ema(close, 50)
    rsi = float(rsi14(close).iloc[-1])

    above25 = close.iloc[-1] > e25.iloc[-1]
    above50 = close.iloc[-1] > e50.iloc[-1]
    slope25 = (e25.iloc[-1] / e25.iloc[-6] - 1.0) if np.isfinite(e25.iloc[-6]) else 0.0

    score = 50
    score += 15 if above25 else -10
    score += 10 if above50 else -10
    score += int(np.clip((rsi - 50) * 0.8, -15, 15))
    score += int(np.clip(slope25 * 1000, -10, 10))
    score = int(np.clip(score, 0, 100))

    if len(close) >= 4 and close.iloc[-4] > 0:
        delta_3d = (close.iloc[-1] / close.iloc[-4] - 1.0) * 100.0
    else:
        delta_3d = 0.0

    risk_on = (np.isfinite(futures_pct) and futures_pct >= 1.5)
    return MarketSnapshot(
        score=score,
        score_label=_label(score),
        delta_3d=float(delta_3d),
        futures_pct=float(futures_pct),
        futures_ticker=futures_ticker,
        risk_on=risk_on,
    )
