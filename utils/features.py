# utils/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def hh(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).max()


def ll(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).min()


@dataclass(frozen=True)
class Tech:
    close: float
    open_: float
    atr: float
    atr_pct: float
    rsi: float
    sma20: float
    sma50: float
    sma10: float
    hh20: float
    ll20: float
    vol: float
    vol_ma20: float
    ret20: float


def compute_tech(df: pd.DataFrame) -> Optional[Tech]:
    if df is None or len(df) < 80:
        return None

    close = df["Close"]
    open_ = df["Open"]
    vol = df["Volume"]

    sma10 = sma(close, 10)
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    atr14 = atr(df, 14)
    rsi14 = rsi(close, 14)

    hh20 = hh(close, 20)
    ll20 = ll(close, 20)
    vol_ma20 = vol.rolling(20).mean()

    c = float(close.iloc[-1])
    o = float(open_.iloc[-1])
    a = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else 0.0
    atr_pct = (a / c) if c > 0 else 0.0
    r = float(rsi14.iloc[-1])
    s10 = float(sma10.iloc[-1])
    s20 = float(sma20.iloc[-1])
    s50 = float(sma50.iloc[-1])
    h20 = float(hh20.iloc[-1])
    l20 = float(ll20.iloc[-1])
    v = float(vol.iloc[-1])
    vm20 = float(vol_ma20.iloc[-1]) if not np.isnan(vol_ma20.iloc[-1]) else 0.0
    ret20 = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0

    return Tech(
        close=c,
        open_=o,
        atr=a,
        atr_pct=atr_pct,
        rsi=r,
        sma20=s20,
        sma50=s50,
        sma10=s10,
        hh20=h20,
        ll20=l20,
        vol=v,
        vol_ma20=vm20,
        ret20=ret20,
    )