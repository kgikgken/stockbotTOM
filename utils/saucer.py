from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.util import safe_float


@dataclass
class SaucerCandidate:
    ticker: str
    name: str
    sector: str
    timeframe: str  # 'W' or 'M'
    score: float
    rim: float
    last: float
    entry_price: float


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily OHLCV to weekly/monthly bars."""
    if df is None or df.empty:
        return pd.DataFrame()

    # pandas 3.0+ tightened/removed some legacy offset aliases (e.g. 'M').
    # We use 'M' to mean month-end, so map to the supported equivalent.
    if rule == "M":
        rule = "ME"
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.sort_index()
    o = d["Open"].resample(rule).first()
    h = d["High"].resample(rule).max()
    l = d["Low"].resample(rule).min()
    c = d["Close"].resample(rule).last()
    v = d["Volume"].resample(rule).sum() if "Volume" in d.columns else None
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
    if v is not None:
        out["Volume"] = v
    return out.dropna()


def _saucer_score(close: pd.Series) -> Optional[Dict[str, float]]:
    """Heuristic saucer scoring on a close series (weekly/monthly).

    Requirements (pragmatic, spec-aligned):
      - window length >= 40
      - U-shape (convex quadratic fit)
      - bottom occurs in the middle band (avoid monotonic trends)
      - last close is near rim (completed / close to completion)

    Returns dict with {score, rim, last} or None.
    """
    if close is None:
        return None
    close = close.dropna().astype(float)
    if len(close) < 40:
        return None

    # Use a rolling window ending at last
    w = close.tail(78) if len(close) >= 78 else close.copy()
    n = len(w)
    if n < 40:
        return None

    y = w.values
    t = np.linspace(-1.0, 1.0, n)
    # normalize y for stability
    y0 = float(np.median(y))
    if not np.isfinite(y0) or y0 <= 0:
        return None
    yn = y / y0

    # quadratic fit: yn = a t^2 + b t + c
    a, b, c = np.polyfit(t, yn, 2)

    if not np.isfinite(a) or a <= 0:
        return None

    # vertex location: t* = -b/(2a)
    t_vertex = -b / (2 * a)
    # require bottom in the middle band
    if not (-0.4 <= t_vertex <= 0.4):
        return None

    rim = float(np.max(y))
    last = float(y[-1])
    if not (np.isfinite(rim) and np.isfinite(last) and rim > 0):
        return None

    progress = last / rim  # close to 1 => near rim
    if progress < 0.90:
        return None

    depth = (rim - float(np.min(y))) / rim
    if depth < 0.10:
        return None  # too shallow, not a meaningful base

    # convexity strength + progress + depth
    score = 0.0
    score += float(np.clip(a, 0.0, 0.5)) * 2.0
    score += (progress - 0.90) / 0.10  # 0..1
    score += float(np.clip(depth, 0.10, 0.40) - 0.10) / 0.30  # 0..1

    score = float(np.clip(score, 0.0, 5.0))
    return {"score": score, "rim": rim, "last": last}


def scan_saucers(
    ohlc_map: Dict[str, pd.DataFrame],
    universe_df: pd.DataFrame,
    ticker_col: str,
    max_n: int = 5,
) -> List[dict]:
    """Return top saucer candidates (weekly/monthly) as list of dicts for reporting."""
    out: List[SaucerCandidate] = []
    if universe_df is None or universe_df.empty:
        return []

    for _, row in universe_df.iterrows():
        ticker = str(row.get(ticker_col, "")).strip()
        if not ticker:
            continue
        df = ohlc_map.get(ticker)
        if df is None or df.empty or len(df) < 260:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        # Weekly
        w = _resample_ohlc(df, "W-FRI")
        sw = _saucer_score(w["Close"]) if not w.empty else None
        if sw:
            rim = float(sw["rim"])
            last = float(sw["last"])
            entry_price = rim  # rim breakout / touch
            out.append(SaucerCandidate(ticker, name, sector, "W", float(sw["score"]), rim, last, entry_price))

        # Monthly
        m = _resample_ohlc(df, "M")
        sm = _saucer_score(m["Close"]) if not m.empty else None
        if sm:
            rim = float(sm["rim"])
            last = float(sm["last"])
            entry_price = rim
            out.append(SaucerCandidate(ticker, name, sector, "M", float(sm["score"]) + 0.5, rim, last, entry_price))

    if not out:
        return []

    out.sort(key=lambda x: (x.score, x.timeframe, x.ticker), reverse=True)
    out = out[: max_n]

    # serialize
    return [
        {
            "ticker": c.ticker,
            "name": c.name,
            "sector": c.sector,
            "timeframe": c.timeframe,
            "score": float(c.score),
            "rim": float(c.rim),
            "last": float(c.last),
            "entry_price": float(c.entry_price),
        }
        for c in out
    ]
