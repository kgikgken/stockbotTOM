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
    # Central limit price used for execution (midpoint of entry band).
    entry_price: Optional[float] = None
    # RR at TP1 (used as expected R basis for CAGR contribution score).
    rr_tp1: Optional[float] = None

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

def structure_sl_tp(df: pd.DataFrame, entry_price: float, atr: float, macro_on: bool) -> Tuple[float, float, float, float, float]:
    """Compute SL/TP targets and a conservative expected holding days.

    Spec alignment:
      - 期待RはTP1基準（RR(TP1)）で固定する。
      - TP1は「構造目標」（直近の重要高値/ブレイク基準）を優先し、無理に同一RRへクリップしない。
      - 想定日数はTP1までの距離(ATR換算)から保守的に算出し、1〜7日にクランプする。
      - TP2は参考表示のみ（スコアには使わない）。
    """
    atr = float(atr) if np.isfinite(atr) and atr > 0 else 1.0

    # --- SL: 直近の押し安値を優先（なければATR基準） ---
    low_10 = safe_float(df["Low"].tail(10).min(), np.nan)
    sl = entry_price - 1.1 * atr
    if np.isfinite(low_10):
        sl = min(float(low_10), entry_price - 0.9 * atr)

    risk = max(1e-6, entry_price - sl)

    # --- TP1: 構造目標（20日高値 / 60日高値 / ブレイクライン） ---
    high_20 = safe_float(df["High"].tail(20).max(), np.nan)
    high_60 = safe_float(df["High"].tail(60).max(), np.nan) if len(df) >= 60 else high_20
    if not np.isfinite(high_20):
        high_20 = entry_price + 2.0 * atr
    if not np.isfinite(high_60):
        high_60 = high_20

    # ブレイク寄りかどうか（60d高値に近い）
    near_breakout = bool(entry_price >= 0.975 * float(high_60))

    tp1_raw = float(high_60 if near_breakout else high_20)

    # 最低限の到達可能性（小さすぎるTPは捨てる）
    tp1_floor = entry_price + max(0.8 * atr, 0.9 * risk)
    tp1 = float(max(tp1_raw, tp1_floor))

    # 上限（極端な目標は抑制）
    tp1_cap = entry_price + 6.0 * atr
    tp1 = float(min(tp1, tp1_cap))

    # --- TP2: 参考表示のみ ---
    tp2 = float(min(tp1 + 0.8 * risk, entry_price + 8.0 * atr))
    if tp2 <= tp1:
        tp2 = float(tp1 + 0.3 * risk)

    rr_tp2 = (tp2 - entry_price) / risk

    # --- 想定日数: TP1距離(ATR)→日数へ（保守的） ---
    # 目標距離が大きいほど日数が伸びる。係数は保守寄り。
    dist_atr = (tp1 - entry_price) / max(1e-6, atr)
    base_days = 0.9 + 0.85 * dist_atr
    # macro_on でもゲートしないが、地合い警戒なら少し保守化
    if macro_on:
        base_days *= 1.10
    expected_days = float(clamp(base_days, 1.0, 7.0))

    return float(sl), float(tp1), float(tp2), float(rr_tp2), float(expected_days)

def build_setup_info(df: pd.DataFrame, macro_on: bool) -> Optional[SetupInfo]:
    det = detect_setup(df)
    if det is None:
        return None
    setup, tier_hint = det
    if setup == "NONE":
        return None

    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)

    atr_s = atr14(df)
    atr_last = safe_float(atr_s.iloc[-1], np.nan)
    if not np.isfinite(atr_last) or atr_last <= 0:
        return None

    entry_low, entry_high, atr_used, breakout_line = entry_band(df, setup)
    if not (np.isfinite(entry_low) and np.isfinite(entry_high) and entry_high > entry_low):
        return None

    entry_price = float((entry_low + entry_high) / 2.0)

    sl, tp1, tp2, rr_tp2, expected_days = structure_sl_tp(df, entry_price, atr_last, macro_on)

    risk = max(1e-6, entry_price - sl)
    rr_tp1 = float((tp1 - entry_price) / risk)

    # 表示RRはTP1基準（期待R）
    rr_display = rr_tp1

    # 回転効率の素朴な初期値（最終はrr_ev側でCAGR寄与度として扱う）
    rday = float(rr_display / max(0.5, expected_days))

    # tier（仕様：A1-Strong=0, A1=1, A2/B=2）
    if setup == "A1-Strong":
        tier = 0
    elif setup == "A1":
        tier = 1
    else:
        tier = 2

    # 簡易トレンド・押し目品質（確率推定に使う）
    trend_strength = _trend_strength(c, ma20, ma50)
    pullback_quality = _pullback_quality(c, ma20, ma50, atr_last, setup)

    gu = gu_flag(df, atr_last)

    return SetupInfo(
        setup=str(setup),
        tier=int(tier),
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        sl=float(sl),
        tp2=float(tp2),
        tp1=float(tp1),
        rr=float(rr_display),
        expected_days=float(expected_days),
        rday=float(rday),
        trend_strength=float(trend_strength),
        pullback_quality=float(pullback_quality),
        gu=bool(gu),
        breakout_line=(float(breakout_line) if breakout_line is not None else None),
        entry_price=float(entry_price),
        rr_tp1=float(rr_tp1),
    )

