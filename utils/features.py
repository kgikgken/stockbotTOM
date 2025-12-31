from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _atr14(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 20:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(14).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else float("nan")


def _rsi14(close: pd.Series) -> float:
    if close is None or len(close) < 20:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]
    rs = float(avg_gain) / (float(avg_loss) + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if not np.isfinite(rsi):
        return 50.0
    return float(rsi)


def compute_features(hist: pd.DataFrame) -> Dict:
    df = hist.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    c = float(close.iloc[-1])
    atr = _atr14(df)

    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else c
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else c
    ma20_prev5 = float(close.rolling(20).mean().iloc[-6]) if len(close) >= 26 else ma20
    ma20_slope5 = (ma20 - ma20_prev5) / (abs(ma20_prev5) + 1e-9)

    rsi = _rsi14(close)

    # turnover (approx JPY)
    turnover = close * vol
    adv20 = float(turnover.rolling(20).mean().iloc[-1]) if len(turnover) >= 20 else float("nan")

    # ATR%
    atr_pct = float(atr / c * 100.0) if np.isfinite(atr) and c > 0 else float("nan")

    # HH20 and HH60
    hh20 = float(high.rolling(20).max().iloc[-2]) if len(high) >= 22 else float(high.max())
    hh60 = float(high.rolling(60).max().iloc[-2]) if len(high) >= 62 else float(high.max())

    # Volume MA20
    vol_ma20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float("nan")
    vol_last = float(vol.iloc[-1]) if len(vol) else float("nan")

    # GU check needs prev close too
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else c
    open_last = float(open_.iloc[-1]) if len(open_) else c

    return {
        "close": c,
        "open": open_last,
        "prev_close": prev_close,
        "atr": float(atr) if np.isfinite(atr) else float("nan"),
        "atr_pct": float(atr_pct) if np.isfinite(atr_pct) else float("nan"),
        "ma20": ma20,
        "ma50": ma50,
        "ma20_slope5": float(ma20_slope5),
        "rsi": float(rsi),
        "adv20": float(adv20) if np.isfinite(adv20) else float("nan"),
        "hh20": float(hh20),
        "hh60": float(hh60),
        "vol_last": float(vol_last) if np.isfinite(vol_last) else float("nan"),
        "vol_ma20": float(vol_ma20) if np.isfinite(vol_ma20) else float("nan"),
    }