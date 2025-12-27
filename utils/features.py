from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.util import sma, rsi14, atr, last, slope_pct

@dataclass(frozen=True)
class Features:
    price: float
    atr: float
    atr_pct: float
    ma20: float
    ma50: float
    ma10: float
    ma5: float
    rsi: float
    slope20_5d: float
    hh20: float
    vol_ma20: float
    turnover_ma20: float
    rs20: float

def compute_features(hist: pd.DataFrame, index_close: Optional[pd.Series] = None) -> Optional[Features]:
    if hist is None or len(hist) < 80:
        return None
    df = hist.copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    a = atr(df, 14)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    rsi = rsi14(close)

    price = last(close)
    atrv = last(a)
    if not (np.isfinite(atrv) and atrv > 0):
        atrv = max(price * 0.015, 1.0)
    atr_pct = (atrv / price) if (np.isfinite(price) and price > 0) else np.nan

    hh20 = float(close.rolling(20).max().iloc[-1]) if len(close) >= 20 else price
    vol_ma20 = float((vol).rolling(20).mean().iloc[-1]) if len(vol) >= 20 else np.nan
    turnover_ma20 = float((close * vol).rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
    slope20_5d = slope_pct(ma20, 5)

    rs20 = 0.0
    if len(close) >= 21:
        st = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        idx_ret = 0.0
        if index_close is not None and len(index_close) >= 21:
            idx_ret = float(index_close.iloc[-1] / index_close.iloc[-21] - 1.0)
        rs20 = st - idx_ret

    return Features(
        price=float(price),
        atr=float(atrv),
        atr_pct=float(atr_pct if np.isfinite(atr_pct) else 0.02),
        ma20=float(last(ma20)),
        ma50=float(last(ma50)),
        ma10=float(last(ma10)),
        ma5=float(last(ma5)),
        rsi=float(last(rsi)),
        slope20_5d=float(slope20_5d),
        hh20=float(hh20),
        vol_ma20=float(vol_ma20),
        turnover_ma20=float(turnover_ma20),
        rs20=float(rs20),
    )
