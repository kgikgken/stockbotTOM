from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.util import sma, rsi14, atr14, adv20, atr_pct_last, safe_float, clamp

@dataclass
class SetupInfo:
    setup: str
    tier: int
    ticker: Optional[str] = None
    sector: Optional[str] = None
    entry_low: float
    entry_high: float
    sl: float
    tp2: float
    tp1: float
    rr: float
    expected_days: float
    rday: float
    trend_strength: float
    pullback_quality: float
    gu: bool
    breakout_line: Optional[float] = None

    @property
    def entry_price(self) -> float:
        """Central limit price used for execution (中央指値)."""
        return float((self.entry_low + self.entry_high) / 2.0)

    @property
    def rr_tp1(self) -> float:
        """RR to TP1 in R units (TP1到達時R)."""
        denom = (self.entry_price - float(self.sl))
        if denom <= 0:
            return 0.0
        return float((float(self.tp1) - self.entry_price) / denom)

    @property
    def expected_r(self) -> float:
        """Expected R (固定): TP1基準のRR (分割利確はスコア外)."""
        return float(self.rr_tp1)

def _trend_strength(c: pd.Series, ma20: pd.Series, ma50: pd.Series) -> float:
    c_last = safe_float(c.iloc[-1], np.nan)
    m20 = safe_float(ma20.iloc[-1], np.nan)
    m50 = safe_float(ma50.iloc[-1], np.nan)
    if not (np.isfinite(c_last) and np.isfinite(m20) and np.isfinite(m50)):
        return 0.0

    slope = safe_float(ma20.pct_change(fill_method=None).iloc[-1], 0.0)

    if c_last > m20 > m50:
        pos = 1.0
    elif c_last > m20:
        pos = 0.7
    elif m20 > m50:
        pos = 0.5
    else:
        pos = 0.3

    ang = clamp((slope / 0.004), -1.0, 1.5)
    base = 0.85 + 0.20 * pos + 0.10 * ang
    return float(clamp(base, 0.80, 1.20))

def _pullback_quality(c: pd.Series, ma20: pd.Series, ma50: pd.Series, atr: float, setup: str) -> float:
    c_last = safe_float(c.iloc[-1], np.nan)
    m20 = safe_float(ma20.iloc[-1], np.nan)
    m50 = safe_float(ma50.iloc[-1], np.nan)
    if not (np.isfinite(c_last) and np.isfinite(m20) and np.isfinite(m50) and np.isfinite(atr) and atr > 0):
        return 0.0

    if setup in ("A1-Strong", "A1"):
        dist = abs(c_last - m20) / atr
        q = 1.15 - clamp(dist, 0.0, 1.6) * 0.18
    elif setup == "A2":
        target = (m20 + m50) / 2.0
        dist = abs(c_last - target) / atr
        q = 1.08 - clamp(dist, 0.0, 2.0) * 0.12
    else:
        q = 1.00

    return float(clamp(q, 0.80, 1.20))

def detect_setup(df: pd.DataFrame) -> Tuple[str, int]:
    if df is None or df.empty or len(df) < 120:
        return "NONE", 9

    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    rsi = rsi14(c)

    c_last = safe_float(c.iloc[-1], np.nan)
    m20 = safe_float(ma20.iloc[-1], np.nan)
    m50 = safe_float(ma50.iloc[-1], np.nan)
    r = safe_float(rsi.iloc[-1], np.nan)
    slope = safe_float(ma20.pct_change(fill_method=None).iloc[-1], 0.0)

    if not (np.isfinite(c_last) and np.isfinite(m20) and np.isfinite(m50) and np.isfinite(r)):
        return "NONE", 9

    if c_last > m20 > m50 and slope > 0:
        if (0.45 <= r/100.0 <= 0.58) and slope >= 0.002:
            return "A1-Strong", 0
        if 40 <= r <= 60:
            return "A1", 1

    if c_last > m50 and slope > -0.001 and 35 <= r <= 60:
        return "A2", 2

    if len(c) >= 25:
        hh20 = float(c.tail(21).max())
        if c_last >= hh20 * 0.997 and slope > 0:
            return "B", 2

    return "NONE", 9

def entry_band(df: pd.DataFrame, setup: str) -> Tuple[float, float, float, Optional[float]]:
    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    a = atr14(df)
    atr = safe_float(a.iloc[-1], np.nan)
    if not np.isfinite(atr) or atr <= 0:
        atr = safe_float(c.iloc[-1], 0.0) * 0.015

    m20 = safe_float(ma20.iloc[-1], safe_float(c.iloc[-1], 0.0))
    m50 = safe_float(ma50.iloc[-1], safe_float(c.iloc[-1], 0.0))
    breakout_line = None

    if setup in ("A1-Strong", "A1"):
        k = 0.4 if setup == "A1-Strong" else 0.5
        lo = m20 - k * atr
        hi = m20 + k * atr
    elif setup == "A2":
        center = (m20 + m50) / 2.0
        lo = center - 0.6 * atr
        hi = center + 0.6 * atr
    elif setup == "B":
        hh20 = float(c.tail(21).max())
        breakout_line = hh20
        lo = hh20 - 0.3 * atr
        hi = hh20 + 0.3 * atr
    else:
        lo = hi = safe_float(c.iloc[-1], 0.0)

    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    lo = max(lo, 1.0)
    hi = max(hi, lo + 0.1)
    return lo, hi, float(atr), (float(breakout_line) if breakout_line is not None else None)

def gu_flag(df: pd.DataFrame, atr: float) -> bool:
    if df is None or df.empty or len(df) < 2:
        return False
    o = safe_float(df["Open"].iloc[-1], np.nan)
    pc = safe_float(df["Close"].iloc[-2], np.nan)
    if not (np.isfinite(o) and np.isfinite(pc) and np.isfinite(atr) and atr > 0):
        return False
    return bool(o > pc + 1.0 * atr)

def liquidity_filters(df: pd.DataFrame, price_min=200.0, price_max=15000.0, adv_min=200e6, atrpct_min=1.5):
    price = safe_float(df["Close"].iloc[-1], np.nan)
    adv = adv20(df)
    atrp = atr_pct_last(df)
    ok = True
    if not (np.isfinite(price) and price_min <= price <= price_max):
        ok = False
    if not (np.isfinite(adv) and adv >= adv_min):
        ok = False
    if not (np.isfinite(atrp) and atrp >= atrpct_min):
        ok = False
    return ok, float(price), float(adv), float(atrp)

def structure_sl_tp(df: pd.DataFrame, entry_mid: float, atr: float, macro_on: bool):
    lookback = 12
    low = float(df["Low"].astype(float).tail(lookback).min())
    sl1 = entry_mid - 1.2 * atr
    sl = min(sl1, low - 0.1 * atr)

    sl = min(sl, entry_mid * (1.0 - 0.02))
    sl = max(sl, entry_mid * (1.0 - 0.10))

    risk = max(entry_mid - sl, 0.01)

    rr_target = 2.6
    hi_window = 60 if len(df) >= 60 else len(df)
    high_60 = float(df["Close"].astype(float).tail(hi_window).max())
    tp2_raw = entry_mid + rr_target * risk
    tp2 = min(tp2_raw, high_60 * 0.995, entry_mid * (1.0 + 0.35))

    tp2_min = entry_mid + 2.0 * risk
    if tp2 < tp2_min:
        tp2 = tp2_min

    tp2_max = entry_mid + 3.5 * risk
    tp2 = min(tp2, tp2_max)

    if macro_on:
        tp2 = entry_mid + (tp2 - entry_mid) * 0.85

    tp1 = entry_mid + 1.5 * risk
    rr = (tp2 - entry_mid) / risk
    exp_days = (tp2 - entry_mid) / max(atr, 1e-6)

    return float(sl), float(tp1), float(tp2), float(rr), float(exp_days)

def build_setup_info(df: pd.DataFrame, macro_on: bool) -> SetupInfo:
    setup, tier = detect_setup(df)
    lo, hi, atr, breakout_line = entry_band(df, setup)
    entry_mid = (lo + hi) / 2.0

    gu = gu_flag(df, atr)
    sl, tp1, tp2, rr, exp_days = structure_sl_tp(df, entry_mid, atr, macro_on=macro_on)
    rday = rr / max(exp_days, 1e-6)

    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    ts = _trend_strength(c, ma20, ma50)
    pq = _pullback_quality(c, ma20, ma50, atr, setup)

    return SetupInfo(
        setup=setup,
        tier=int(tier),
        entry_low=float(lo),
        entry_high=float(hi),
        sl=float(sl),
        tp2=float(tp2),
        tp1=float(tp1),
        rr=float(rr),
        expected_days=float(exp_days),
        rday=float(rday),
        trend_strength=float(ts),
        pullback_quality=float(pq),
        gu=bool(gu),
        breakout_line=breakout_line,
    )
