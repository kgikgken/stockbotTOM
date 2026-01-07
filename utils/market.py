# ============================================
# utils/market.py
# 市場（地合い）スコアと相場判断
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import clamp


@dataclass
class MarketContext:
    market_score: float
    delta_3d: float
    regime: str  # "上昇基調" / "中立" / "弱含み"
    macro_risk: bool


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _fetch_index(symbol: str = "^TOPX", period: str = "1y") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        return df
    return pd.DataFrame()


def calc_market_score(symbol: str = "^TOPX") -> Tuple[float, float, str]:
    """
    MarketScore(0-100), Δ3d, regime_text を返す
    """
    df = _fetch_index(symbol=symbol, period="1y")
    if df.empty:
        return 50.0, 0.0, "中立"

    close = df["Close"].dropna()
    if len(close) < 60:
        return 50.0, 0.0, "中立"

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    rsi14 = _rsi(close, 14)

    c = float(close.iloc[-1])
    c3 = float(close.iloc[-4]) if len(close) >= 4 else float(close.iloc[-1])
    delta_3d = float((c / c3 - 1.0) * 100.0)

    trend = 0.0
    if c > float(sma50.iloc[-1]):
        trend += 20.0
    if float(sma20.iloc[-1]) > float(sma50.iloc[-1]):
        trend += 20.0

    mom = 0.0
    rsi = float(rsi14.iloc[-1])
    if rsi >= 55:
        mom += 15.0
    elif rsi <= 45:
        mom -= 10.0

    if delta_3d >= 0:
        mom += 10.0
    else:
        mom -= 10.0

    score = 50.0 + trend + mom
    score = clamp(score, 0.0, 100.0)

    if score >= 60:
        regime = "上昇基調"
    elif score >= 45:
        regime = "中立"
    else:
        regime = "弱含み"

    return float(score), float(delta_3d), regime


def build_market_context(macro_risk: bool) -> MarketContext:
    score, d3, regime = calc_market_score("^TOPX")
    return MarketContext(
        market_score=score,
        delta_3d=d3,
        regime=regime,
        macro_risk=macro_risk,
    )