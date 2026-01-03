from __future__ import annotations

import numpy as np
import pandas as pd

from utils.features import sma, atr


def calc_in_band(df: pd.DataFrame, setup_type: str) -> tuple[float, float, float]:
    """
    IN_center と帯 (low, high)
    A: SMA20 ± 0.5ATR
    B: HH20 ± 0.3ATR
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)

    c = float(close.iloc[-1])
    a = atr(df, 14)
    if not np.isfinite(a) or a <= 0:
        a = max(c * 0.01, 1.0)

    if setup_type == "B" and len(high) >= 25:
        hh20 = float(high.iloc[-21:-1].max())
        center = hh20
        w = 0.3 * a
    else:
        center = sma(close, 20)
        if not np.isfinite(center):
            center = c
        w = 0.5 * a

    low = float(center - w)
    high_ = float(center + w)
    return float(center), low, high_