from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.features import FeaturePack, estimate_pwin
from utils.util import safe_float


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return safe_float(v)


@dataclass
class RRResult:
    stop: float
    tp1: float
    tp2: float
    rr: float
    expected_days: float
    r_per_day: float
    ev: float
    adj_ev: float
    reason: str


def compute_rr_ev(
    hist: pd.DataFrame,
    fp: FeaturePack,
    entry_center: float,
    setup_type: str,
    sector_rank: int | None,
    regime_multiplier: float,
    rday_cap_when_weak: float,
    mkt_score: int,
) -> RRResult:
    """
    固定RR禁止（RRは結果）
    Stop:
      A：STOP = IN_low − 0.7ATR（≒ IN_center - 1.2ATR）
      B：STOP = BreakLine − 1.0ATR
      + 直近安値が近い場合 min(...)
    Target:
      TP2 = min(レジスタンス, measured_move, IN + 3.5ATR)
      TP1 = IN + 1.5R
    RR = (TP2-IN)/R
    EV = Pwin*RR - (1-Pwin)*1
    AdjEV = EV * regime_multiplier
    """
    df = hist.copy()
    close = df["Close"].astype(float)

    atr = fp.atr if np.isfinite(fp.atr) and fp.atr > 0 else _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(fp.close * 0.01, 1.0)

    in_center = float(entry_center)

    # swing low
    lookback = 12
    swing_low = safe_float(df["Low"].astype(float).tail(lookback).min())

    # stop
    if setup_type == "B":
        stop_base = in_center - 1.0 * atr
    else:
        stop_base = in_center - 1.2 * atr

    # structure stop: swing_low buffer
    stop = min(stop_base, swing_low - 0.2 * atr) if np.isfinite(swing_low) else stop_base

    # clamp stop distance
    r = in_center - stop
    if not np.isfinite(r) or r <= 0:
        return RRResult(0, 0, 0, 0, 99, 0, -9, -9, "STOP算出失敗")
    if r / in_center < 0.02:
        stop = in_center * (1 - 0.02)
        r = in_center - stop
    if r / in_center > 0.10:
        stop = in_center * (1 - 0.10)
        r = in_center - stop

    # resistance (60d high)
    hi_window = 60 if len(close) >= 60 else len(close)
    resistance = safe_float(close.tail(hi_window).max())

    # measured move: (20d high - 20d low) added to entry
    mm_win = 20 if len(close) >= 20 else len(close)
    mm_hi = safe_float(df["High"].astype(float).tail(mm_win).max())
    mm_lo = safe_float(df["Low"].astype(float).tail(mm_win).min())
    measured_move = in_center + max(0.0, (mm_hi - mm_lo))

    tp2_candidates = []
    if np.isfinite(resistance) and resistance > 0:
        tp2_candidates.append(resistance * 0.995)
    if np.isfinite(measured_move) and measured_move > 0:
        tp2_candidates.append(measured_move)
    tp2_candidates.append(in_center + 3.5 * atr)

    tp2 = float(min(tp2_candidates))
    if tp2 <= in_center:
        tp2 = in_center + 2.2 * r  # fail-safe

    rr = (tp2 - in_center) / r

    # tp1 = in + 1.5R
    tp1 = in_center + 1.5 * r

    # expected days: (TP2-IN)/(k*ATR)
    k = 1.0
    expected_days = (tp2 - in_center) / (k * atr) if atr > 0 else 99.0
    expected_days = float(np.clip(expected_days, 1.0, 7.0))
    r_per_day = float(rr / expected_days) if expected_days > 0 else 0.0

    # Pwin & EV
    pwin = estimate_pwin(fp, sector_rank)
    ev = float(pwin * rr - (1.0 - pwin) * 1.0)
    adj_ev = float(ev * regime_multiplier)

    # 地合い弱い時の速度上限制御（仕様③）
    if mkt_score < 55 and r_per_day > rday_cap_when_weak:
        return RRResult(
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            rr=float(rr),
            expected_days=float(expected_days),
            r_per_day=float(r_per_day),
            ev=float(ev),
            adj_ev=float(adj_ev),
            reason="R/day過大(地合い弱)→WATCH_ONLY",
        )

    return RRResult(
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        rr=float(rr),
        expected_days=float(expected_days),
        r_per_day=float(r_per_day),
        ev=float(ev),
        adj_ev=float(adj_ev),
        reason="ok",
    )