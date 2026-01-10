from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import pandas as pd

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    v = float(rsi.iloc[-1])
    return v if np.isfinite(v) else np.nan

def detect_setup(hist: pd.DataFrame) -> Tuple[str, Dict[str, float], bool]:
    """Return (setup_type, anchors, gu_flag).
    anchors include: sma20, sma50, atr, hh20, entry_mid, entry_lo, entry_hi
    """
    if hist is None or len(hist) < 80:
        return "NONE", {}, False

    df = hist.copy()
    c = df["Close"].astype(float)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    sma20 = float(c.rolling(20).mean().iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])
    sma20_prev = float(c.rolling(20).mean().iloc[-6]) if len(c) >= 26 else sma20
    slope20 = (sma20 / (sma20_prev + 1e-9) - 1.0)

    atr = _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(float(c.iloc[-1]) * 0.015, 1.0)

    rsi = _rsi(c, 14)
    price = float(c.iloc[-1])
    prev_close = float(c.iloc[-2])
    open_today = float(o.iloc[-1])

    # GU: Open > prev_close + 1ATR
    gu = bool(open_today > prev_close + atr)

    # HH20
    hh20 = float(h.tail(21).max()) if len(h) >= 21 else float(h.max())
    vol_avg20 = float(v.rolling(20).mean().iloc[-2]) if len(v) >= 22 else float(v.mean())
    vol_today = float(v.iloc[-1]) if np.isfinite(v.iloc[-1]) else np.nan

    # Volume pullback pattern (A1): vol down into pullback, then rebound on last bar
    vol_pullback_ok = False
    try:
        vol3 = v.tail(4).astype(float)
        close3 = c.tail(4).astype(float)
        # last 3 vols decreasing and last close >= prev close
        if len(vol3) >= 4:
            vol_pullback_ok = bool(vol3.iloc[-2] < vol3.iloc[-3] and vol3.iloc[-1] >= vol3.iloc[-2] and close3.iloc[-1] >= close3.iloc[-2])
    except Exception:
        vol_pullback_ok = False

    setup = "NONE"
    anchors: Dict[str, float] = {"sma20": sma20, "sma50": sma50, "atr": atr, "hh20": hh20}

    # A1
    if price > sma20 > sma50 and slope20 > 0 and np.isfinite(rsi) and 40 <= rsi <= 60:
        if abs(price - sma20) <= 0.5 * atr:
            setup = "A1"
    # A2
    if setup == "NONE":
        if price > sma50 and (sma20 - 0.5 * atr) >= sma50 and (sma50 - 0.3 * atr) <= price <= (sma20 + 0.2 * atr) and np.isfinite(rsi) and 35 <= rsi <= 65:
            setup = "A2"
    # B (breakout)
    if setup == "NONE":
        if price >= hh20 * 1.001 and np.isfinite(vol_today) and np.isfinite(vol_avg20) and vol_avg20 > 0 and (vol_today >= 1.5 * vol_avg20):
            setup = "B"

    # Entry band per setup
    if setup in ("A1", "A2"):
        entry_mid = sma20
        entry_lo = sma20 - 0.5 * atr
        entry_hi = sma20 + 0.5 * atr
    elif setup == "B":
        entry_mid = hh20
        entry_lo = hh20 - 0.3 * atr
        entry_hi = hh20 + 0.3 * atr
    else:
        entry_mid = price
        entry_lo = price
        entry_hi = price

    anchors.update({"entry_mid": float(entry_mid), "entry_lo": float(entry_lo), "entry_hi": float(entry_hi), "rsi": float(rsi) if np.isfinite(rsi) else np.nan, "slope20": float(slope20)})
    # Optional: strengthen A1 with vol pullback, but do NOT downgrade if missing (data noise)
    anchors["vol_pullback_ok"] = 1.0 if vol_pullback_ok else 0.0

    return setup, anchors, gu
