from __future__ import annotations

import numpy as np
import pandas as pd

from .rr import atr14, hh20

def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan

def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma20_slope5"] = df["ma20"].pct_change(5, fill_method=None)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["turnover"] = c * v
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()
    df["vol_ma20"] = v.rolling(20).mean()
    return df

def universe_ok(hist: pd.DataFrame, close_min=200, close_max=15000, adv_min=200_000_000) -> tuple[bool, dict]:
    if hist is None or len(hist) < 120:
        return False, {}
    df = add_indicators(hist)
    c = _last(df["Close"])
    adv = _last(df["turnover_avg20"])
    atr = atr14(hist)
    atr_pct = float(atr / c) if np.isfinite(atr) and np.isfinite(c) and c > 0 else np.nan

    ok = True
    if not (np.isfinite(c) and close_min <= c <= close_max):
        ok = False
    if not (np.isfinite(adv) and adv >= adv_min):
        ok = False
    if not (np.isfinite(atr_pct) and atr_pct >= 0.015):
        ok = False
    if np.isfinite(atr_pct) and atr_pct >= 0.06:
        ok = False

    return ok, {"close": c, "adv20": adv, "atr": atr, "atr_pct": atr_pct}

def setup_type(hist: pd.DataFrame) -> tuple[str, dict]:
    df = add_indicators(hist)
    c = _last(df["Close"])
    o = _last(df["Open"]) if "Open" in df.columns else np.nan
    pc = float(df["Close"].astype(float).iloc[-2]) if len(df) >= 2 else np.nan

    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    slope5 = _last(df["ma20_slope5"])
    rsi = _last(df["rsi14"])

    vol = _last(df["Volume"]) if "Volume" in df.columns else np.nan
    vol_ma20 = _last(df["vol_ma20"])

    atr = atr14(hist)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(c * 0.01, 1.0)

    gu = bool(np.isfinite(o) and np.isfinite(pc) and (o > pc + 1.0 * atr))

    cond_a = (
        np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50) and
        c > ma20 > ma50 and
        np.isfinite(slope5) and slope5 > 0 and
        np.isfinite(rsi) and 40 <= rsi <= 62 and
        abs(c - ma20) <= 0.8 * atr
    )

    h20 = hh20(hist)
    cond_b = (
        np.isfinite(c) and np.isfinite(h20) and c > h20 and
        np.isfinite(vol) and np.isfinite(vol_ma20) and vol_ma20 > 0 and vol >= 1.5 * vol_ma20
    )

    if cond_a:
        return "A", {"gu": gu, "atr": atr, "ma20": ma20, "ma50": ma50, "rsi": rsi}
    if cond_b:
        return "B", {"gu": gu, "atr": atr, "h20": h20, "rsi": rsi}
    return "NONE", {"gu": gu, "atr": atr}

def in_zone(hist: pd.DataFrame, stype: str) -> tuple[float, float, float]:
    df = add_indicators(hist)
    c = _last(df["Close"])
    atr = atr14(hist)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(c * 0.01, 1.0)

    if stype == "A":
        center = float(_last(df["ma20"]))
        low = center - 0.5 * atr
        high = center + 0.5 * atr
    else:
        center = float(hh20(hist))
        low = center - 0.3 * atr
        high = center + 0.3 * atr
    return float(center), float(low), float(high)

def action_label(price_now: float, in_center: float, atr: float, gu_flag: bool) -> tuple[str, float]:
    if gu_flag:
        return "WATCH_ONLY", 9.99
    if not (np.isfinite(price_now) and np.isfinite(in_center) and np.isfinite(atr) and atr > 0):
        return "WATCH_ONLY", 9.99
    dev = abs(price_now - in_center) / atr
    if dev > 0.8:
        return "WATCH_ONLY", float(dev)
    if dev > 0.3:
        return "LIMIT_WAIT", float(dev)
    return "EXEC_NOW", float(dev)

def pwin_proxy(hist: pd.DataFrame, stype: str, sector_rank: int | None, mkt_score: int) -> float:
    df = add_indicators(hist)
    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    slope5 = _last(df["ma20_slope5"])
    rsi = _last(df["rsi14"])
    adv = _last(df["turnover_avg20"])

    atr = atr14(hist)
    atr_pct = (atr / c) if np.isfinite(atr) and np.isfinite(c) and c > 0 else 0.03

    ts = 0.0
    if np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50):
        if c > ma20 > ma50:
            ts += 0.35
        elif c > ma20:
            ts += 0.20
        elif ma20 > ma50:
            ts += 0.10
    if np.isfinite(slope5):
        ts += float(np.clip(slope5 / 0.03, -0.1, 0.2))

    rs = 0.05
    if np.isfinite(rsi):
        rs = 0.20 if 42 <= rsi <= 58 else 0.15 if 40 <= rsi <= 62 else 0.05

    liq = 0.0
    if np.isfinite(adv):
        liq = float(np.clip((adv - 2e8) / 1.8e9, 0.0, 0.15))

    vola = 0.10
    if np.isfinite(atr_pct):
        if 0.015 <= atr_pct <= 0.045:
            vola = 0.15
        elif atr_pct > 0.06:
            vola = 0.03

    sec = 0.0
    if sector_rank is not None and sector_rank >= 1:
        sec = float(np.clip((6 - sector_rank) / 10.0, 0.0, 0.10))

    st = 0.05 if stype == "A" else 0.03
    mk = float(np.clip((mkt_score - 50) / 200.0, -0.05, 0.05))

    p = 0.35 + ts + rs + liq + vola + sec + st + mk
    return float(np.clip(p, 0.05, 0.75))

def regime_multiplier(mkt_score: int, delta3d: int, has_event_near: bool) -> float:
    mult = 1.0
    if mkt_score >= 60 and delta3d >= 0:
        mult *= 1.05
    if delta3d <= -5:
        mult *= 0.70
    if has_event_near:
        mult *= 0.75
    return float(np.clip(mult, 0.5, 1.1))

def ev_from(pwin: float, rr: float) -> float:
    return float(pwin * rr - (1.0 - pwin) * 1.0)
