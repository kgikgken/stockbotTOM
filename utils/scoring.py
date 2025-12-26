# utils/scoring.py
from __future__ import annotations
import numpy as np
import pandas as pd

def universe_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["price"] >= 200) & (df["price"] <= 15000)]
    df = df[df["adv20"] >= 2e8]
    return df.reset_index(drop=True)


def detect_setup(hist: pd.DataFrame) -> str | None:
    c = hist["Close"]
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    atr = (hist["High"] - hist["Low"]).rolling(14).mean()

    if c.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]:
        if abs(c.iloc[-1] - ma20.iloc[-1]) <= 0.8 * atr.iloc[-1]:
            return "A"
    if c.iloc[-1] >= c.rolling(20).max().iloc[-1]:
        return "B"
    return None


def calc_entry_zone(hist: pd.DataFrame, setup: str) -> dict:
    c = hist["Close"]
    atr = (hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1]
    open_ = hist["Open"].iloc[-1]
    prev = c.iloc[-2]

    if setup == "A":
        center = c.rolling(20).mean().iloc[-1]
        width = 0.5 * atr
    else:
        center = c.rolling(20).max().iloc[-1]
        width = 0.3 * atr

    gu = open_ > prev + atr
    return {
        "center": center,
        "low": center - width,
        "high": center + width,
        "atr": atr,
        "gu": gu,
        "close": c.iloc[-1],
    }


def calc_action(zone: dict) -> str:
    if zone["gu"]:
        return "WATCH_ONLY"
    dist = abs(zone["close"] - zone["center"]) / zone["atr"]
    if dist <= 0.5:
        return "EXEC_NOW"
    if dist <= 0.8:
        return "LIMIT_WAIT"
    return "WATCH_ONLY"


def calc_pwin(hist: pd.DataFrame, sector: str, sectors: list, gu: bool) -> float:
    c = hist["Close"]
    ma20 = c.rolling(20).mean()
    slope = (ma20.iloc[-1] - ma20.iloc[-6]) / ma20.iloc[-6]
    base = 0.35
    base += min(max(slope * 10, 0), 0.15)
    for i, (s, _) in enumerate(sectors):
        if s == sector:
            base += max(0.1 - i * 0.02, 0)
    if gu:
        base *= 0.5
    return float(np.clip(base, 0.2, 0.7))