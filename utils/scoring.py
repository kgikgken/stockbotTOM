from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float("nan")


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)

    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def _add(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"].astype(float)
    v = d["Volume"].astype(float) if "Volume" in d.columns else pd.Series(np.nan, index=d.index)

    d["ma20"] = c.rolling(20).mean()
    d["ma50"] = c.rolling(50).mean()
    d["ma10"] = c.rolling(10).mean()
    d["hh20"] = c.rolling(20).max()
    d["vol20"] = v.rolling(20).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    d["rsi14"] = 100 - (100 / (1 + rs))

    d["ret20"] = c.pct_change(20, fill_method=None)
    d["ret5"] = c.pct_change(5, fill_method=None)
    return d


def trend_gate(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 80:
        return False
    d = _add(hist)
    c = _last(d["Close"])
    ma20 = _last(d["ma20"])
    ma50 = _last(d["ma50"])
    if not (np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50)):
        return False
    if not (ma20 > ma50 and c > ma50):
        return False
    return True


def detect_setup_type(hist: pd.DataFrame) -> str:
    if hist is None or len(hist) < 80:
        return "N"

    d = _add(hist)
    c = _last(d["Close"])
    ma20 = _last(d["ma20"])
    ma50 = _last(d["ma50"])
    ma10 = _last(d["ma10"])
    hh20 = _last(d["hh20"])
    rsi = _last(d["rsi14"])

    atr = _atr(hist, 14)
    if not np.isfinite(atr) or atr <= 0:
        return "N"

    slope20 = float(d["ma20"].pct_change(5, fill_method=None).iloc[-1]) if len(d) >= 30 else 0.0
    dist_ma20 = abs(c - ma20)

    cond_a = (
        np.isfinite(ma20) and np.isfinite(ma50) and np.isfinite(c) and
        c > ma20 > ma50 and
        slope20 > 0 and
        dist_ma20 <= 0.8 * atr and
        (np.isfinite(rsi) and 38 <= rsi <= 60)
    )
    if cond_a:
        return "A"

    vol = hist["Volume"].astype(float) if "Volume" in hist.columns else pd.Series(np.nan, index=hist.index)
    vol20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 25 else float("nan")
    vol_last = float(vol.iloc[-1]) if len(vol) else float("nan")
    cond_b = (
        np.isfinite(hh20) and np.isfinite(c) and c >= hh20 * 0.999 and
        np.isfinite(vol_last) and np.isfinite(vol20) and vol20 > 0 and vol_last >= 1.5 * vol20 and
        np.isfinite(ma10) and c >= ma10
    )
    if cond_b:
        return "B"

    return "N"


def calc_in_zone(hist: pd.DataFrame, setup: str) -> Dict[str, float]:
    d = _add(hist)
    c = _last(d["Close"])
    ma20 = _last(d["ma20"])
    hh20 = _last(d["hh20"])

    atr = _atr(hist, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(float(c) * 0.01 if np.isfinite(c) else 1.0, 1.0)

    if setup == "A":
        center = float(ma20)
        lower = center - 0.5 * atr
        upper = center + 0.5 * atr
        basis = "MA20±0.5ATR"
    else:
        center = float(hh20)
        lower = center - 0.3 * atr
        upper = center + 0.3 * atr
        basis = "HH20±0.3ATR"

    return {"center": float(center), "lower": float(lower), "upper": float(upper), "atr": float(atr), "basis": basis}


def estimate_pwin(hist: pd.DataFrame, sector_rank: int) -> float:
    if hist is None or len(hist) < 80:
        return 0.0

    d = _add(hist)
    c = d["Close"].astype(float)
    ret = c.pct_change(fill_method=None)

    ma20 = _last(d["ma20"])
    ma50 = _last(d["ma50"])
    close = _last(c)
    rsi = _last(d["rsi14"])
    ret20 = _last(d["ret20"])
    ret5 = _last(d["ret5"])
    vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 25 else float("nan")

    sc = 0.0

    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 > ma50:
            sc += 0.35
        elif close > ma50:
            sc += 0.20

    if np.isfinite(rsi):
        if 50 <= rsi <= 62:
            sc += 0.15
        elif 45 <= rsi < 50:
            sc += 0.08
        elif rsi > 70:
            sc -= 0.10

    if np.isfinite(ret20):
        sc += float(np.clip(ret20 * 2.0, -0.10, 0.18))

    if np.isfinite(ret5):
        sc += float(np.clip(ret5 * 1.5, -0.08, 0.12))

    if sector_rank <= 5:
        sc += (6 - sector_rank) * 0.02
    else:
        sc -= 0.05

    if np.isfinite(vola20):
        if vola20 >= 0.035:
            sc -= 0.12
        elif vola20 <= 0.015:
            sc += 0.05

    p = 0.35 + sc
    return float(np.clip(p, 0.15, 0.65))
