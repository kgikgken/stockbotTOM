from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.util import safe_float


@dataclass
class SaucerHit:
    ticker: str
    name: str
    sector: str
    tf: str  # 'D' | 'W' | 'M'
    rim_price: float
    progress: float  # 0-1, 1 = breakout at rim
    depth: float     # 0-1, cup depth ratio
    score: float     # ranking score (internal)


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out = ohlc.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    out = out.dropna()
    return out


def _saucer_score(
    close: pd.Series,
    *,
    min_len: int,
    lookback: int,
    min_progress: float,
    min_depth: float,
    vertex_band: float,
) -> Optional[Dict[str, float]]:
    """Return saucer metrics or None.

    Heuristic saucer definition (auditable / deterministic):
      - Rim = max(close) in lookback window
      - Bottom = min(close) in lookback window
      - Depth = (rim - bottom) / rim  (needs to be 'deep enough')
      - Progress = close_last / rim   (needs to be close to rim)
      - Vertex location: bottom should be around middle of window (avoid straight V)
    """
    if close is None:
        return None
    c = close.dropna().astype(float)
    if len(c) < min_len:
        return None

    c = c.tail(lookback) if len(c) > lookback else c
    if len(c) < min_len:
        return None

    rim = float(np.nanmax(c.values))
    bottom = float(np.nanmin(c.values))
    last = float(c.iloc[-1])

    if not (np.isfinite(rim) and np.isfinite(bottom) and np.isfinite(last)) or rim <= 0:
        return None
    if bottom <= 0:
        return None

    depth = (rim - bottom) / rim
    if depth < min_depth:
        return None

    progress = last / rim
    if progress < min_progress:
        return None

    # Vertex position should not be too close to the edges (avoid sharp V / straight decline)
    idx_bottom = int(np.nanargmin(c.values))
    mid = (len(c) - 1) / 2.0
    vertex_pos = abs(idx_bottom - mid) / max(1.0, mid)  # 0 at center, 1 at edge
    if vertex_pos > vertex_band:
        return None

    # Smoothness proxy: penalize extremely jagged cups
    # Use normalized std of returns as roughness.
    rets = c.pct_change(fill_method=None).dropna()
    rough = float(np.nanstd(rets.values)) if len(rets) >= 10 else 0.0

    # Ranking score: prefer deeper + closer to rim + smoother
    score = (progress * 1.5) + (depth * 1.2) - (rough * 8.0)

    return {
        "rim": rim,
        "bottom": bottom,
        "last": last,
        "depth": float(depth),
        "progress": float(progress),
        "score": float(score),
    }


def scan_saucers(
    ohlc_map: Dict[str, pd.DataFrame],
    uni: pd.DataFrame,
    tcol: str,
    *,
    max_each: int = 5,
) -> Dict[str, List[Dict]]:
    """Scan saucers on D/W/M. Returns dict for report.

    - Daily: stricter and longer window (avoid noisy short cups)
    - Weekly/Monthly: structural saucers
    """
    out: Dict[str, List[SaucerHit]] = {"D": [], "W": [], "M": []}
    if uni is None or uni.empty:
        return {"D": [], "W": [], "M": []}

    for _, row in uni.iterrows():
        ticker = str(row.get(tcol, "")).strip()
        if not ticker:
            continue
        df = ohlc_map.get(ticker)
        if df is None or df.empty or len(df) < 180:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        # Daily (strict)
        met = _saucer_score(
            df["Close"],
            min_len=160,
            lookback=260,
            min_progress=0.94,
            min_depth=0.12,
            vertex_band=0.55,
        )
        if met:
            out["D"].append(
                SaucerHit(
                    ticker=ticker,
                    name=name,
                    sector=sector,
                    tf="D",
                    rim_price=float(met["rim"]),
                    progress=float(met["progress"]),
                    depth=float(met["depth"]),
                    score=float(met["score"]),
                )
            )

        # Weekly
        try:
            w = _resample(df, "W-FRI")
        except Exception:
            w = None
        if w is not None and not w.empty and len(w) >= 60:
            met = _saucer_score(
                w["Close"],
                min_len=50,
                lookback=104,  # ~2y
                min_progress=0.92,
                min_depth=0.15,
                vertex_band=0.60,
            )
            if met:
                out["W"].append(
                    SaucerHit(
                        ticker=ticker,
                        name=name,
                        sector=sector,
                        tf="W",
                        rim_price=float(met["rim"]),
                        progress=float(met["progress"]),
                        depth=float(met["depth"]),
                        score=float(met["score"]),
                    )
                )

        # Monthly
        try:
            m = _resample(df, "M")
        except Exception:
            m = None
        if m is not None and not m.empty and len(m) >= 36:
            met = _saucer_score(
                m["Close"],
                min_len=30,
                lookback=120,  # up to 10y
                min_progress=0.90,
                min_depth=0.18,
                vertex_band=0.70,
            )
            if met:
                out["M"].append(
                    SaucerHit(
                        ticker=ticker,
                        name=name,
                        sector=sector,
                        tf="M",
                        rim_price=float(met["rim"]),
                        progress=float(met["progress"]),
                        depth=float(met["depth"]),
                        score=float(met["score"]),
                    )
                )

    # Sort and truncate
    ret: Dict[str, List[Dict]] = {}
    for tf in ("D", "W", "M"):
        hits = out[tf]
        hits.sort(key=lambda x: (x.score, x.progress, x.depth, x.rim_price, x.ticker), reverse=True)
        hits = hits[: max_each]
        ret[tf] = [
            {
                "ticker": h.ticker,
                "name": h.name,
                "sector": h.sector,
                "tf": h.tf,
                "rim": float(h.rim_price),
                "progress": float(h.progress),
                "depth": float(h.depth),
                "score": float(h.score),
            }
            for h in hits
        ]
    return ret
