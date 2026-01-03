# utils/scoring.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ============================================================
# Helpers
# ============================================================
def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _sma(series: pd.Series, n: int) -> float:
    if series is None or len(series) < n:
        return np.nan
    return float(series.rolling(n).mean().iloc[-1])


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    if df is None or len(df) < n + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()],
        axis=1
    ).max(axis=1)
    v = tr.rolling(n).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


# ============================================================
# Indicators
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"].astype(float)
    v = out["Volume"].astype(float) if "Volume" in out.columns else pd.Series(np.nan, index=out.index)

    out["sma20"] = c.rolling(20).mean()
    out["sma50"] = c.rolling(50).mean()
    out["sma10"] = c.rolling(10).mean()

    # RSI14
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    out["rsi14"] = 100 - (100 / (1 + rs))

    # ATR
    out["atr14"] = _atr(out, 14)

    # Vol metrics
    out["vol_ma20"] = v.rolling(20).mean()
    out["turnover"] = c * v
    out["turnover_ma20"] = out["turnover"].rolling(20).mean()

    # Highs
    out["hh20"] = c.rolling(20).max()

    # Slope (trend strength)
    out["sma20_slope"] = out["sma20"].pct_change(5)

    return out


# ============================================================
# Setup判定（A/Bのみ）
# ============================================================
def detect_setup(df: pd.DataFrame) -> str:
    """
    戻り値: "A", "B", "-"
    """
    if df is None or len(df) < 60:
        return "-"

    c = _last(df["Close"])
    sma20 = _last(df["sma20"])
    sma50 = _last(df["sma50"])
    rsi = _last(df["rsi14"])
    atr = _last(df["atr14"])
    hh20 = _last(df["hh20"])
    slope = _last(df["sma20_slope"])

    if not all(np.isfinite(x) for x in [c, sma20, sma50, rsi, atr]):
        return "-"

    # --- Setup A: トレンド押し目 ---
    if (
        c > sma20 > sma50
        and slope > 0
        and abs(c - sma20) <= 0.8 * atr
        and 40 <= rsi <= 62
    ):
        return "A"

    # --- Setup B: ブレイク（厳選） ---
    vol = _last(df["Volume"])
    vol_ma20 = _last(df["vol_ma20"])
    if (
        c >= hh20
        and np.isfinite(vol)
        and np.isfinite(vol_ma20)
        and vol >= 1.5 * vol_ma20
    ):
        return "B"

    return "-"


# ============================================================
# Entry Zone / GU / Action
# ============================================================
def calc_entry_action(df: pd.DataFrame, setup: str) -> Dict:
    """
    IN帯・GU・乖離率・Action を返す
    """
    c = _last(df["Close"])
    o = float(df["Open"].iloc[-1]) if "Open" in df.columns else c
    prev_c = float(df["Close"].iloc[-2]) if len(df) >= 2 else c
    atr = _last(df["atr14"])
    sma20 = _last(df["sma20"])
    hh20 = _last(df["hh20"])

    if not np.isfinite(atr) or atr <= 0:
        return {"action": "WATCH_ONLY"}

    if setup == "A":
        center = sma20
        band = 0.5 * atr
    elif setup == "B":
        center = hh20
        band = 0.3 * atr
    else:
        return {"action": "WATCH_ONLY"}

    low = center - band
    high = center + band

    # GU判定
    gu_flag = bool(o > prev_c + 1.0 * atr)

    # 乖離率
    dist = abs(c - center) / atr

    # Action
    if gu_flag or dist > 0.8:
        action = "WATCH_ONLY"
    elif low <= c <= high:
        action = "EXEC_NOW"
    else:
        action = "LIMIT_WAIT"

    return {
        "in_center": float(center),
        "in_low": float(low),
        "in_high": float(high),
        "gu_flag": gu_flag,
        "dist_atr": float(dist),
        "action": action,
    }


# ============================================================
# RR / EV / 速度
# ============================================================
def calc_rr_ev_speed(df: pd.DataFrame, entry: float, setup: str, mkt_mult: float) -> Dict:
    """
    RR >= 2.2 / EV >= 0.4R / R/day >= 0.5 を想定
    """
    atr = _last(df["atr14"])
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
        return {"ok": False}

    # Stop
    if setup == "A":
        stop = entry - 1.2 * atr
    else:  # B
        stop = entry - 1.0 * atr

    risk = entry - stop
    if risk <= 0:
        return {"ok": False}

    # Target
    tp2 = entry + 3.0 * risk
    r = (tp2 - entry) / risk

    # Pwin proxy（簡易・代理）
    pwin = 0.38
    pwin += np.clip((_last(df["sma20_slope"]) * 50), -0.05, 0.10)
    pwin = float(np.clip(pwin, 0.25, 0.55))

    ev = pwin * r - (1 - pwin) * 1.0
    adj_ev = ev * mkt_mult

    # 速度
    expected_days = (tp2 - entry) / (1.0 * atr)
    r_per_day = r / max(expected_days, 1e-6)

    ok = bool(
        r >= 2.2
        and ev >= 0.4 * r
        and expected_days <= 5
        and r_per_day >= 0.5
    )

    return {
        "ok": ok,
        "rr": float(r),
        "ev": float(ev),
        "adj_ev": float(adj_ev),
        "expected_days": float(expected_days),
        "r_per_day": float(r_per_day),
        "stop": float(stop),
        "tp2": float(tp2),
    }