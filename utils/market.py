from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from utils.util import download_history_bulk, safe_float


MARKET_TICKERS = {
    "nikkei": "^N225",
    "topix": "^TOPX",
    "jpn_etf": "1306.T",
}

FUTURES_TICKERS = ["ES=F", "NQ=F"]


def _one_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 220:
        return float("nan")
    close = df["Close"].astype(float)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    last = close.iloc[-1]
    high52 = close.rolling(252).max().iloc[-1]
    parts = [
        1.0 if last > ma20.iloc[-1] else 0.0,
        1.0 if last > ma50.iloc[-1] else 0.0,
        1.0 if last > ma200.iloc[-1] else 0.0,
        1.0 if ma20.iloc[-1] > ma50.iloc[-1] else 0.0,
        1.0 if ma50.iloc[-1] > ma200.iloc[-1] else 0.0,
        1.0 if ma200.iloc[-1] >= ma200.shift(20).iloc[-1] else 0.0,
        1.0 if (high52 - last) / high52 * 100.0 <= 12.0 else 0.0,
    ]
    return float(np.mean(parts) * 100.0)


def market_score(ohlc_map_override: Dict[str, pd.DataFrame] | None = None) -> Dict[str, object]:
    data = ohlc_map_override or download_history_bulk(list(MARKET_TICKERS.values()), period="18mo")
    available = []
    lines = []
    for name, ticker in MARKET_TICKERS.items():
        df = data.get(ticker)
        if df is None or df.empty:
            continue
        score = _one_score(df)
        if not np.isfinite(score):
            continue
        available.append(score)
        close = safe_float(df["Close"].iloc[-1])
        lines.append(f"{name}:{score:.0f} ({close:,.0f})")

    if not available:
        return {
            "score": 55,
            "label": "neutral",
            "lines": ["market data unavailable"],
        }
    score = int(round(float(np.mean(available))))
    if score >= 70:
        label = "strong"
    elif score >= 55:
        label = "normal"
    elif score >= 40:
        label = "soft"
    else:
        label = "weak"
    return {
        "score": score,
        "label": label,
        "lines": lines,
    }


def futures_risk_on(ohlc_map_override: Dict[str, pd.DataFrame] | None = None) -> tuple[bool, float]:
    data = ohlc_map_override or download_history_bulk(FUTURES_TICKERS, period="3mo")
    changes: list[float] = []
    for ticker in FUTURES_TICKERS:
        df = data.get(ticker)
        if df is None or df.empty or len(df) < 2:
            continue
        close = df["Close"].astype(float)
        chg = float((close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0)
        if np.isfinite(chg):
            changes.append(chg)
    if not changes:
        return False, 0.0
    mean_chg = float(np.mean(changes))
    return mean_chg >= 0.0, mean_chg
