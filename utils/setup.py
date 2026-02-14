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
    # Liquidity/volatility telemetry (used by reach-prob heuristic).
    adv20: Optional[float] = None
    atrp: Optional[float] = None
    # Central limit price used for execution (midpoint of entry band).
    entry_price: Optional[float] = None
    # RR at TP1 (used as expected R basis for CAGR contribution score).
    rr_tp1: Optional[float] = None
    # --- Quality/structure telemetry for main setups (used for screening and reach-prob refinement)
    # 20日騰落（%）
    ret20: Optional[float] = None
    # 出来高比（直近5本平均 / 直近20本平均）
    vol_ratio: Optional[float] = None
    # ボラ収縮（ATR%直近5本平均 / 直近20本平均）  <1 が収縮
    atr_contr: Optional[float] = None
    # ギャップ頻度（直近20本で|Gap|>1.2% の比率）
    gap_freq: Optional[float] = None
    # レンジ収縮（(H-L)/Close の直近5本平均 / 直近20本平均）
    range_contr: Optional[float] = None


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


def _ret_n(close: pd.Series, n: int = 20) -> float:
    """n本騰落（%）"""
    try:
        n = int(n)
    except Exception:
        n = 20
    if close is None or len(close) < n + 1:
        return np.nan
    base = safe_float(close.iloc[-(n + 1)], np.nan)
    last = safe_float(close.iloc[-1], np.nan)
    if not (np.isfinite(base) and np.isfinite(last) and base > 0):
        return np.nan
    return float((last / base - 1.0) * 100.0)


def _vol_ratio(df: pd.DataFrame, short: int = 5, long: int = 20) -> float:
    """出来高比（直近short平均 / 直近long平均）"""
    if df is None or df.empty or "Volume" not in df.columns:
        return np.nan
    v = df["Volume"].astype(float)
    if len(v) < max(short, long) + 1:
        return np.nan
    vs = safe_float(v.tail(short).mean(), np.nan)
    vl = safe_float(v.tail(long).mean(), np.nan)
    if not (np.isfinite(vs) and np.isfinite(vl) and vl > 0):
        return np.nan
    return float(vs / vl)


def _atrp_contr(df: pd.DataFrame, short: int = 5, long: int = 20) -> float:
    """ATR%収縮（直近short平均 / 直近long平均）"""
    if df is None or df.empty or len(df) < max(short, long) + 30:
        # ATR14のウォームアップも必要
        return np.nan
    a = atr14(df)
    c = df["Close"].astype(float)
    atrp = (a / (c + 1e-9)) * 100.0
    if len(atrp) < max(short, long):
        return np.nan
    s = safe_float(atrp.tail(short).mean(), np.nan)
    l = safe_float(atrp.tail(long).mean(), np.nan)
    if not (np.isfinite(s) and np.isfinite(l) and l > 0):
        return np.nan
    return float(s / l)


def _range_contr(df: pd.DataFrame, short: int = 5, long: int = 20) -> float:
    """レンジ収縮（(H-L)/Close の直近short平均 / 直近long平均）"""
    if df is None or df.empty or len(df) < max(short, long) + 1:
        return np.nan
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    rng = ((h - l).abs() / (c + 1e-9)) * 100.0
    s = safe_float(rng.tail(short).mean(), np.nan)
    lg = safe_float(rng.tail(long).mean(), np.nan)
    if not (np.isfinite(s) and np.isfinite(lg) and lg > 0):
        return np.nan
    return float(s / lg)


def _gap_freq(df: pd.DataFrame, window: int = 20, thresh: float = 0.012) -> float:
    """ギャップ頻度（直近window本で|Gap|>thresh の比率）"""
    if df is None or df.empty or len(df) < window + 2:
        return np.nan
    o = df["Open"].astype(float)
    pc = df["Close"].astype(float).shift(1)
    gap = ((o - pc).abs() / (pc + 1e-9))
    return float((gap.tail(window) > float(thresh)).mean())

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

def structure_sl_tp(
    df: pd.DataFrame,
    entry_price: float,
    atr: float,
    macro_on: bool,
    setup: str = "A1",
):
    """Compute structural SL/TP targets.

    Latest spec alignment:
      - Expected R uses RR(TP1) (TP1 is structural).
      - TP2 is reference-only (not used for scoring).
      - expected_days is conservative and tied to TP1 distance and ATR.

    Returns (sl, tp1, tp2, rr_tp2, expected_days)
    """
    atr = float(atr) if np.isfinite(atr) and atr > 0 else max(1.0, float(entry_price) * 0.015)

    # SL: prefer recent swing low, otherwise ATR-based.
    low_10 = safe_float(df["Low"].astype(float).tail(10).min(), np.nan)
    sl = float(entry_price - 1.0 * atr)
    if np.isfinite(low_10):
        sl = float(min(low_10, entry_price - 0.8 * atr))

    # Safety clamps: avoid too-tight / too-wide SL.
    sl = float(min(sl, entry_price * (1.0 - 0.01)))
    sl = float(max(sl, entry_price * (1.0 - 0.12)))

    risk = max(1e-6, entry_price - sl)

    high_20 = float(df["High"].astype(float).tail(20).max())
    high_60 = float(df["High"].astype(float).tail(60).max()) if len(df) >= 60 else high_20

    # Heuristic breakout context: near 60d high.
    near_breakout = entry_price >= (0.97 * high_60)
    tp1_raw = high_60 if near_breakout else high_20

    # Clamp TP1 to realistic zone.
    tp1_min = entry_price + 0.8 * risk
    tp1_max = min(entry_price + 2.5 * risk, entry_price + 4.0 * atr)
    tp1 = float(np.clip(tp1_raw, tp1_min, tp1_max))

    # TP2: reference only.
    tp2_raw = tp1 + 0.7 * risk
    tp2_max = entry_price + 5.5 * atr
    tp2 = float(np.clip(tp2_raw, tp1 + 0.2 * risk, tp2_max))

    if macro_on:
        # During macro caution, keep TP2 modest (display-only) to avoid overreach.
        tp2 = float(entry_price + (tp2 - entry_price) * 0.90)

    rr_tp2 = float((tp2 - entry_price) / risk)

    # Conservative expected days based on TP1 distance + setup floor (to prevent unrealistic 1.0d).
    base_days = float((tp1 - entry_price) / max(1e-6, atr))
    expected_days = float(base_days * 1.15)
    floor_map = {
        "A1-Strong": 2.2,
        "A1": 2.4,
        "A2": 2.8,
        "B": 1.8,
    }
    expected_days = max(expected_days, float(floor_map.get(setup, 2.5)))
    if macro_on:
        expected_days += 0.2
    expected_days = float(clamp(expected_days, 1.0, 7.0))

    return float(sl), float(tp1), float(tp2), float(rr_tp2), float(expected_days)

def build_setup_info(df: pd.DataFrame, macro_on: bool, entry_override: float | None = None) -> SetupInfo:
    setup, tier = detect_setup(df)
    lo, hi, atr, breakout_line = entry_band(df, setup)
    entry_price = float(entry_override) if entry_override is not None and np.isfinite(entry_override) and entry_override > 0 else (lo + hi) / 2.0

    gu = gu_flag(df, atr)
    sl, tp1, tp2, rr_tp2, exp_days = structure_sl_tp(df, entry_price, atr, macro_on=macro_on, setup=setup)
    # RR at TP1 defines expected R basis (TP1 fixed).
    denom = max(entry_price - sl, 1e-9)
    rr_tp1 = max((tp1 - entry_price) / denom, 0.0)
    rday = rr_tp1 / max(exp_days, 1e-6)

    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    ts = _trend_strength(c, ma20, ma50)
    pq = _pullback_quality(c, ma20, ma50, atr, setup)

    # --- quality telemetry (screening)
    ret20 = _ret_n(c, 20)
    vol_ratio = _vol_ratio(df, 5, 20)
    atr_contr = _atrp_contr(df, 5, 20)
    gap_freq = _gap_freq(df, 20, 0.012)
    range_contr = _range_contr(df, 5, 20)

    return SetupInfo(
        entry_price=entry_price,
        rr_tp1=rr_tp1,
        setup=setup,
        tier=int(tier),
        entry_low=float(lo),
        entry_high=float(hi),
        sl=float(sl),
        tp2=float(tp2),
        tp1=float(tp1),
        rr=float(rr_tp1),
        expected_days=float(exp_days),
        rday=float(rday),
        trend_strength=float(ts),
        pullback_quality=float(pq),
        gu=bool(gu),
        breakout_line=breakout_line,
        ret20=float(ret20) if np.isfinite(ret20) else None,
        vol_ratio=float(vol_ratio) if np.isfinite(vol_ratio) else None,
        atr_contr=float(atr_contr) if np.isfinite(atr_contr) else None,
        gap_freq=float(gap_freq) if np.isfinite(gap_freq) else None,
        range_contr=float(range_contr) if np.isfinite(range_contr) else None,
    )


def build_position_info(df: pd.DataFrame, entry_price: float, macro_on: bool) -> Optional[SetupInfo]:
    """Build SetupInfo for an existing position.

    最新仕様（監査OS）方針：
      - ポジション欄も新規と同じ「RR(TP1)・到達確率・期待R×到達確率・CAGR寄与度(/日)・想定日数」で監査する。
      - 現在の形（setup）が検出できない日でも、監査のための指標は必ず算出する。
        → setup='POS' として扱い、SL/TP1/TP2 は entry_price を基準に構造的に再計算する。
    """
    if df is None or df.empty or len(df) < 60 or not np.isfinite(entry_price) or entry_price <= 0:
        return None

    setup, tier = detect_setup(df)
    if setup == "NONE":
        setup = "POS"
        tier = 9

    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)

    a = atr14(df)
    atr_last = safe_float(a.iloc[-1], np.nan)
    if not np.isfinite(atr_last) or atr_last <= 0:
        return None

    # positions: entry band is not used (execution already happened)
    entry_low = entry_high = float(entry_price)

    sl, tp1, tp2, rr_tp2, expected_days = structure_sl_tp(
        df,
        float(entry_price),
        float(atr_last),
        macro_on=bool(macro_on),
        setup=str(setup),
    )

    risk = max(1e-6, float(entry_price) - float(sl))
    rr_tp1 = float((float(tp1) - float(entry_price)) / risk)
    rr_display = rr_tp1

    rday = float(rr_display / max(0.5, float(expected_days)))

    trend_strength = _trend_strength(c, ma20, ma50)
    pullback_quality = _pullback_quality(c, ma20, ma50, float(atr_last), setup)

    # --- quality telemetry (positions)
    ret20 = _ret_n(c, 20)
    vol_ratio = _vol_ratio(df, 5, 20)
    atr_contr = _atrp_contr(df, 5, 20)
    gap_freq = _gap_freq(df, 20, 0.012)
    range_contr = _range_contr(df, 5, 20)

    info = SetupInfo(
        setup=setup,
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
        gu=False,
        breakout_line=None,
        adv20=float(adv20(df)),
        atrp=float(atr_pct_last(df)),
        entry_price=float(entry_price),
        rr_tp1=float(rr_tp1),
        ret20=float(ret20) if np.isfinite(ret20) else None,
        vol_ratio=float(vol_ratio) if np.isfinite(vol_ratio) else None,
        atr_contr=float(atr_contr) if np.isfinite(atr_contr) else None,
        gap_freq=float(gap_freq) if np.isfinite(gap_freq) else None,
        range_contr=float(range_contr) if np.isfinite(range_contr) else None,
    )
    return info

