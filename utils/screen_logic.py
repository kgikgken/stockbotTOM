from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from utils.setup import sma, rsi14, atr14, sma_slope_up, hh20, _last
from utils.rr_ev import compute_exit_levels, compute_ev_metrics

def detect_setup(hist: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    df = hist.copy()
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    c = _last(close)

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    rsi = rsi14(close)
    atr = atr14(df)
    if not (np.isfinite(c) and np.isfinite(sma20) and np.isfinite(sma50) and np.isfinite(atr) and atr > 0):
        return "NA", {}

    trend_ok = bool(c > sma20 > sma50 and sma_slope_up(close, 20, 5))

    dist20_atr = abs(c - sma20) / atr if atr > 0 else 999.0
    dist50_atr = abs(c - sma50) / atr if atr > 0 else 999.0

    setup = "NA"
    breakout_line = np.nan

    if trend_ok and np.isfinite(rsi) and (40 <= rsi <= 60) and dist20_atr <= 0.8:
        setup = "A1"
        entry_mid = sma20
        entry_lo = sma20 - 0.5 * atr
        entry_hi = sma20 + 0.5 * atr
    elif (c > sma50) and np.isfinite(rsi) and (35 <= rsi <= 65) and dist50_atr <= 1.6:
        setup = "A2"
        entry_mid = sma20
        entry_lo = min(sma20 - 0.8 * atr, sma50 - 0.3 * atr)
        entry_hi = sma20 + 0.4 * atr
    else:
        br = hh20(close)
        breakout_line = br
        if np.isfinite(br) and c >= br * 1.002:
            setup = "B"
            entry_mid = br
            entry_lo = br - 0.3 * atr
            entry_hi = br + 0.3 * atr
        else:
            return "NA", {}

    anchors = {
        "entry_mid": float(entry_mid),
        "entry_lo": float(entry_lo),
        "entry_hi": float(entry_hi),
        "atr": float(atr),
        "rsi": float(rsi) if np.isfinite(rsi) else np.nan,
        "sma20": float(sma20),
        "sma50": float(sma50),
        "breakout_line": float(breakout_line) if np.isfinite(breakout_line) else np.nan,
        "last_close": float(c),
        "last_open": float(_last(open_)),
    }
    return setup, anchors

def gu_flag(anchors: Dict[str, float]) -> bool:
    try:
        o = float(anchors.get("last_open", np.nan))
        c = float(anchors.get("last_close", np.nan))
        atr = float(anchors.get("atr", np.nan))
        if not (np.isfinite(o) and np.isfinite(c) and np.isfinite(atr) and atr > 0):
            return False
        return bool(o > c + 1.0 * atr)
    except Exception:
        return False

def score_candidate(hist: pd.DataFrame, setup: str, anchors: Dict[str, float], *, mkt_score: int, macro_on: bool) -> Optional[Dict]:
    if setup == "NA":
        return None
    atr = float(anchors["atr"])
    entry_mid = float(anchors["entry_mid"])

    exits = compute_exit_levels(hist, entry_mid, atr, macro_on=macro_on)
    rr = float(exits["rr"])
    sl = float(exits["sl"])
    tp1 = float(exits["tp1"])
    tp2 = float(exits["tp2"])

    gu = gu_flag(anchors)
    ev, adjev, expected_days, rday = compute_ev_metrics(setup, rr, atr, entry_mid, tp2, mkt_score, gu)

    return {
        "setup": setup,
        "entry_lo": float(anchors["entry_lo"]),
        "entry_hi": float(anchors["entry_hi"]),
        "entry_mid": float(entry_mid),
        "atr": float(atr),
        "rr": float(rr),
        "ev": float(ev),
        "adjev": float(adjev),
        "expected_days": float(expected_days),
        "rday": float(rday),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "gu": bool(gu),
    }
