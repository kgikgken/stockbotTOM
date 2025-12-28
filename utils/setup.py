from __future__ import annotations

import yfinance as yf
import numpy as np


def judge_setup(ticker: str) -> dict:
    df = yf.download(ticker, period="1y", auto_adjust=True)
    if len(df) < 100:
        return {"valid": False}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    atr = (high - low).rolling(14).mean().iloc[-1]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    rsi_val = rsi.iloc[-1]

    # Setup A（押し目）
    if (
        close.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]
        and abs(close.iloc[-1] - sma20.iloc[-1]) <= 0.8 * atr
        and 40 <= rsi_val <= 62
    ):
        return {
            "valid": True,
            "setup_type": "A",
            "atr": atr,
            "sma20": sma20.iloc[-1],
        }

    # Setup B（ブレイク）
    hh20 = high.rolling(20).max().iloc[-2]
    vol_ma = vol.rolling(20).mean().iloc[-1]

    if close.iloc[-1] > hh20 and vol.iloc[-1] >= 1.5 * vol_ma:
        return {
            "valid": True,
            "setup_type": "B",
            "break_line": hh20,
            "atr": atr,
        }

    return {"valid": False}