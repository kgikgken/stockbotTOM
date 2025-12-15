from __future__ import annotations

import numpy as np
import pandas as pd


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _ma(series: pd.Series, n: int) -> float:
    if series is None or len(series) < n:
        return _last(series)
    return float(series.rolling(n).mean().iloc[-1])


def _turnover_avg(df: pd.DataFrame, n: int = 20) -> float:
    if df is None or len(df) < n:
        return np.nan
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
    tv = close * vol
    v = tv.rolling(n).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def evaluate_runner(hist: pd.DataFrame) -> tuple[str, float]:
    """Runner判定（0-100）"""
    if hist is None or len(hist) < 80:
        return "C", 0.0

    df = hist.copy()
    close = df["Close"].astype(float)
    c = _last(close)
    ma20 = _ma(close, 20)
    ma60 = _ma(close, 60)

    if not (np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma60)):
        return "C", 0.0

    trend = 1.0 if (c > ma20 > ma60) else 0.0

    hi = float(close.tail(60).max()) if len(close) >= 60 else float(close.max())
    dist = (hi / c - 1.0) * 100.0 if c > 0 else 999.0
    near_high = 1.0 if dist <= 6.0 else (0.0 if dist >= 12.0 else (12.0 - dist) / 6.0)

    if len(close) >= 21 and float(close.iloc[-21]) > 0:
        chg20 = (c / float(close.iloc[-21]) - 1.0) * 100.0
    else:
        chg20 = 0.0
    momentum = 1.0 if chg20 >= 8.0 else (0.0 if chg20 <= 2.0 else (chg20 - 2.0) / 6.0)

    t = _turnover_avg(df, 20)
    liq = 1.0 if (np.isfinite(t) and t >= 1e8) else 0.0

    strength = 100.0 * (0.40 * trend + 0.30 * near_high + 0.20 * momentum + 0.10 * liq)
    strength = float(np.clip(strength, 0, 100))

    label = "A2_prebreak" if (trend > 0 and near_high > 0.5 and momentum > 0.5 and liq > 0) else "C"
    return label, strength


def is_al3(c: dict) -> bool:
    return (
        c.get("runner_label") == "A2_prebreak"
        and c.get("in_rank") in ("強IN", "通常IN")
        and float(c.get("rr", 0.0)) >= 2.5
        and float(c.get("ev_r", 0.0)) >= 0.6
    )


def select_al3_top1(cands: list[dict]) -> list[dict]:
    al3 = [c for c in cands if is_al3(c)]
    if not al3:
        return []
    al3.sort(
        key=lambda x: (
            x.get("runner_strength", 0.0) * x.get("ev_r", 0.0) * x.get("rr", 0.0),
            x.get("score", 0.0),
        ),
        reverse=True,
    )
    return [al3[0]]
