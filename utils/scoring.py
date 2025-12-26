import numpy as np
import pandas as pd

def universe_filter(df: pd.DataFrame) -> bool:
    close = df["Close"].iloc[-1]
    if close < 200 or close > 15000:
        return False

    turnover = (df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]
    if turnover < 1e8:
        return False

    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    if atr / close < 0.015 or atr / close > 0.06:
        return False

    return True

def detect_setup(df: pd.DataFrame) -> str | None:
    close = df["Close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    if close.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]:
        return "A"

    hh20 = close.rolling(20).max().iloc[-2]
    if close.iloc[-1] > hh20:
        return "B"

    return None

def calc_in_zone(df: pd.DataFrame, setup: str) -> tuple:
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    close = df["Close"]

    if setup == "A":
        center = close.rolling(20).mean().iloc[-1]
        return center - 0.5 * atr, center + 0.5 * atr

    if setup == "B":
        center = close.rolling(20).max().iloc[-2]
        return center - 0.3 * atr, center + 0.3 * atr

    return None, None

def calc_action(df: pd.DataFrame, in_zone: tuple) -> str:
    close = df["Close"].iloc[-1]
    low, high = in_zone
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]

    if close < low:
        return "LIMIT_WAIT"
    if close > high + 0.8 * atr:
        return "WATCH_ONLY"
    return "EXEC_NOW"

def estimate_pwin(df: pd.DataFrame, sector_rank: int) -> float:
    slope = df["Close"].rolling(20).mean().pct_change().iloc[-1]
    base = 0.35
    base += np.clip(slope * 10, -0.05, 0.10)
    base += max(0, (6 - sector_rank)) * 0.01
    return float(np.clip(base, 0.25, 0.55))