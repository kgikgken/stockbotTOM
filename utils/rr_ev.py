from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

def _swing_low(df: pd.DataFrame, lookback: int = 12) -> float:
    try:
        return float(df["Low"].astype(float).tail(lookback).min())
    except Exception:
        return np.nan

def compute_exit_levels(
    hist: pd.DataFrame,
    entry_mid: float,
    atr: float,
    *,
    macro_on: bool = False,
) -> Dict[str, float]:
    """
    Exit設計（仕様準拠）
    - STOP: IN - 1.2ATR を基本、直近安値が近い場合はそちら優先
    - TP1: +1.5R（部分利確）
    - TP2: +2.0〜3.5R（構造＋抵抗で自然決定）
    - Macro警戒日は「やらない日」ではなく「取り方を変える日」：TP2を控えめに圧縮
    """
    df = hist.copy()
    entry = float(entry_mid)
    atr = float(atr) if np.isfinite(atr) and atr > 0 else max(entry * 0.015, 1.0)

    sl_base = entry - 1.2 * atr
    s_low = _swing_low(df, lookback=12)
    if np.isfinite(s_low):
        sl_struct = s_low - 0.2 * atr
        sl = max(sl_base, sl_struct)
    else:
        sl = sl_base

    sl = float(np.clip(sl, entry * 0.90, entry * 0.98))
    r = max(1e-6, entry - sl)

    tp1 = entry + 1.5 * r

    close = df["Close"].astype(float)
    hi_window = 60 if len(close) >= 60 else len(close)
    high_60 = float(close.tail(hi_window).max()) if hi_window > 0 else entry

    tp2_raw = entry + 2.8 * r
    tp2 = min(entry + 3.5 * r, max(entry + 2.0 * r, min(high_60 * 0.995, tp2_raw)))

    if macro_on:
        tp2 = min(tp2, entry + 2.6 * r)

    rr = (tp2 - entry) / r if r > 0 else 0.0

    return {"sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "rr": float(rr), "r": float(r)}

def pwin_for_setup(setup: str) -> float:
    if setup == "A1":
        return 0.45
    if setup == "A2":
        return 0.40
    if setup == "B":
        return 0.35
    return 0.30

def compute_ev_metrics(
    setup: str,
    rr: float,
    atr: float,
    entry: float,
    tp2: float,
    mkt_score: int,
    macro_on: bool,
    gu: bool,
) -> Tuple[float, float, float, float]:
    rr = float(rr)
    if rr <= 0:
        return -999.0, -999.0, 999.0, 0.0

    p = pwin_for_setup(setup)

    if mkt_score >= 70:
        p += 0.03
    elif mkt_score <= 45:
        p -= 0.03

    # Macro警戒は「減点で候補ゼロ」に直結するので、EV側の一律減点はしない。
    p = float(np.clip(p, 0.20, 0.60))

    ev = p * rr - (1.0 - p)

    adjev = ev
    if gu:
        adjev -= 0.10

    atr = float(atr) if np.isfinite(atr) and atr > 0 else max(float(entry) * 0.015, 1.0)
    expected_days = float((tp2 - entry) / atr) if atr > 0 else 999.0
    expected_days = float(np.clip(expected_days, 0.8, 30.0))

    rday = float(rr / expected_days) if expected_days > 0 else 0.0

    return float(ev), float(adjev), float(expected_days), float(rday)
