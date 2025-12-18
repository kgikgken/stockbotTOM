from __future__ import annotations
import numpy as np
import pandas as pd

def score_stock(hist: pd.DataFrame) -> float | None:
    if len(hist) < 80:
        return None

    close = hist["Close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    rsi = close.diff().apply(lambda x: max(x,0)).rolling(14).mean() / \
          close.diff().abs().rolling(14).mean() * 100

    sc = 0.0
    if close.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]:
        sc += 40
    if ma20.iloc[-1] > ma20.iloc[-6]:
        sc += 20
    if 35 <= rsi.iloc[-1] <= 55:
        sc += 20

    return float(np.clip(sc, 0, 100))

def calc_inout_for_stock(hist: pd.DataFrame):
    close = hist["Close"]
    ma20 = close.rolling(20).mean()
    rsi = close.diff().apply(lambda x: max(x,0)).rolling(14).mean() / \
          close.diff().abs().rolling(14).mean() * 100

    if close.iloc[-1] >= ma20.iloc[-1] and 35 <= rsi.iloc[-1] <= 55:
        return "強IN", None, None
    if close.iloc[-1] >= ma20.iloc[-1]:
        return "通常IN", None, None
    return "様子見", None, None