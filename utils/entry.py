from __future__ import annotations

import yfinance as yf


def judge_entry_action(ticker: str, setup: dict) -> dict:
    df = yf.download(ticker, period="10d", auto_adjust=True)

    open_p = df["Open"].iloc[-1]
    close_p = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    atr = setup["atr"]

    gu = open_p > prev_close + atr

    if setup["setup_type"] == "A":
        center = setup["sma20"]
        width = 0.5 * atr
    else:
        center = setup["break_line"]
        width = 0.3 * atr

    entry_low = center - width
    entry_high = center + width

    dist = abs(close_p - center) / atr

    if gu or dist > 0.8:
        action = "WATCH_ONLY"
    elif entry_low <= close_p <= entry_high:
        action = "EXEC_NOW"
    else:
        action = "LIMIT_WAIT"

    return {
        "entry": center,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "gu": gu,
        "action": action,
    }