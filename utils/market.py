# utils/market.py
from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import DEFAULT_TZ, MarketState, clamp


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _score_trend(close: pd.Series) -> float:
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    last = close.iloc[-1]
    s20 = sma20.iloc[-1]
    s50 = sma50.iloc[-1]

    score = 50.0
    if last > s50:
        score += 10
    if s20 > s50:
        score += 10
    if last > s20:
        score += 5
    return score


def _score_momentum(close: pd.Series) -> float:
    r5 = close.pct_change(5).iloc[-1]
    r20 = close.pct_change(20).iloc[-1]
    rsi = _rsi(close, 14).iloc[-1]
    score = 0.0
    score += clamp(r5 * 200, -10, 10)
    score += clamp(r20 * 150, -10, 10)
    score += clamp((rsi - 50) * 0.4, -10, 10)
    return score


def calc_market_state(ticker: str = "^TOPX", lookback: str = "1y") -> MarketState:
    """
    TOPIX（^TOPX）を基本に地合いを算出。
    """
    try:
        df = yf.download(ticker, period=lookback, interval="1d", auto_adjust=True, progress=False)
        df = df.dropna()
        close = df["Close"]
        if len(close) < 80:
            raise RuntimeError("not enough data")
    except Exception:
        # 取得できない時は中立で落とさない
        return MarketState(
            score=55,
            delta_3d=0,
            regime="neutral",
            macro_caution=False,
            leverage=1.3,
            max_gross=2_600_000,
            no_trade=False,
            reason="(market data unavailable)",
        )

    base = _score_trend(close)
    mom = _score_momentum(close)
    score = clamp(base + mom, 0, 100)

    # 3日変化（点差）
    # ざっくり：3日前時点のスコアとの差分
    close3 = close.iloc[:-3]
    base3 = _score_trend(close3)
    mom3 = _score_momentum(close3)
    score3 = clamp(base3 + mom3, 0, 100)
    delta_3d = float(score - score3)

    # regime
    if score >= 65:
        regime = "bull"
    elif score >= 50:
        regime = "neutral"
    else:
        regime = "bear"

    # NO-TRADE（完全機械化）
    no_trade = (score < 45) or (delta_3d <= -5 and score < 55)
    reason = ""
    if score < 45:
        reason = "MarketScore<45"
    elif delta_3d <= -5 and score < 55:
        reason = "Δ3d<=-5 & MarketScore<55"

    # レバ：地合いに連動（上限は守る）
    if no_trade:
        leverage = 1.0
    else:
        if score >= 70:
            leverage = 2.0
        elif score >= 60:
            leverage = 1.7
        else:
            leverage = 1.3

    # 建玉目安（基準資金2,000,000を想定）
    base_cap = 2_000_000
    max_gross = base_cap * leverage

    return MarketState(
        score=float(score),
        delta_3d=float(delta_3d),
        regime=regime,
        macro_caution=False,
        leverage=float(leverage),
        max_gross=float(max_gross),
        no_trade=bool(no_trade),
        reason=reason,
    )