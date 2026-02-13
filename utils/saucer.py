from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.util import safe_float, adv20, atr_pct_last


@dataclass
class SaucerHit:
    ticker: str
    name: str
    sector: str
    tf: str  # 'D' | 'W' | 'M'
    rim_price: float
    last_price: float
    atrp: float
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
    max_depth: float,
    vertex_band: float,
) -> Optional[Dict[str, float]]:
    """Return saucer metrics or None.

    「ソーサーボトム（カップ/皿）」の深さは浅すぎても(ノイズ)深すぎても(下落トレンド)質が落ちるため、
    timeframe別に「最適レンジ」を採用する。

    depth = (rim - bottom) / rim

    出力する score は「候補の優先度」用途のみ。検知条件は progress / depth / vertex_band で決める。
    """
    c = close.astype(float).dropna()
    if len(c) < max(min_len, lookback + 5):
        return None

    seg = c.tail(lookback)
    rim = float(seg.max())
    bottom = float(seg.min())
    last = float(seg.iloc[-1])

    if rim <= 0:
        return None

    depth = (rim - bottom) / rim
    if not (np.isfinite(depth) and min_depth <= depth <= max_depth):
        return None

    # Progress: how close last is to rim (0..1)
    progress = (last - bottom) / max(1e-9, (rim - bottom))
    if not (np.isfinite(progress) and progress >= min_progress):
        return None

    # Vertex: bottom should be in the first half-ish (avoid V-shaped snapback)
    idx_bottom = int(seg.values.argmin())
    mid = int(len(seg) * 0.55)
    if idx_bottom > mid:
        return None

    # Vertex band: last should not be too far above rim (avoid already extended)
    if last > rim * (1.0 + vertex_band):
        return None

    # Roughness penalty: prefer smoother bowls (lower residual variance vs. rolling mean)
    ma = seg.rolling(5, min_periods=3).mean()
    rough = float(np.nanstd((seg - ma) / rim))

    # Depth preference: maximize closeness to target depth (center of optimal range)
    target = (min_depth + max_depth) / 2.0
    width = max(1e-6, (max_depth - min_depth))
    depth_pen = abs(depth - target) / width  # 0 at target, ~1 at edge

    # Score: prioritize completion first, then depth quality, then smoothness
    score = (progress * 2.0) - (depth_pen * 0.60) - (rough * 8.0)

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

        # Sanity / liquidity guardrails (avoid broken series causing absurd rim prices)
        last_close = safe_float(df["Close"].iloc[-1], np.nan)
        if not np.isfinite(last_close):
            continue
        # JP equities price sanity (very loose). Reject obvious scale bugs (e.g. billions).
        if last_close < 50.0 or last_close > 200000.0:
            continue
        adv = adv20(df)
        atrp = atr_pct_last(df)
        if (not np.isfinite(adv)) or adv < 100e6:
            continue
        if (not np.isfinite(atrp)) or atrp < 0.6:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        # Daily (strict)
        met = _saucer_score(
            df["Close"],
            min_len=160,
            lookback=260,
            min_progress=0.95,
            min_depth=0.35,
            vertex_band=0.55,
                max_depth=0.75,
        )
        if met:
            out["D"].append(
                SaucerHit(
                    ticker=ticker,
                    name=name,
                    sector=sector,
                    tf="D",
                    rim_price=float(met["rim"]),
                    last_price=float(met["last"]),
                    atrp=float(atrp),
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
            atrp_w = atr_pct_last(w)
            met = _saucer_score(
                w["Close"],
                min_len=50,
                lookback=104,  # ~2y
                min_progress=0.95,
                min_depth=0.45,
                vertex_band=0.60,
                max_depth=0.75,
            )
            if met:
                out["W"].append(
                    SaucerHit(
                        ticker=ticker,
                        name=name,
                        sector=sector,
                        tf="W",
                        rim_price=float(met["rim"]),
                        last_price=float(met["last"]),
                        atrp=float(atrp_w),
                        progress=float(met["progress"]),
                        depth=float(met["depth"]),
                        score=float(met["score"]),
                    )
                )

        # Monthly
        try:
            m = _resample(df, "ME")  # pandas>=2 uses month-end "ME"
        except Exception:
            m = None
        if m is not None and not m.empty and len(m) >= 36:
            atrp_m = atr_pct_last(m)
            met = _saucer_score(
                m["Close"],
                min_len=30,
                lookback=120,  # up to 10y
                min_progress=0.92,
                min_depth=0.55,
                vertex_band=0.70,
                max_depth=0.80,
            )
            if met:
                out["M"].append(
                    SaucerHit(
                        ticker=ticker,
                        name=name,
                        sector=sector,
                        tf="M",
                        rim_price=float(met["rim"]),
                        last_price=float(met["last"]),
                        atrp=float(atrp_m),
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
                "last": float(h.last_price),
                "atrp": float(h.atrp),
                "progress": float(h.progress),
                "depth": float(h.depth),
                "score": float(h.score),
            }
            for h in hits
        ]
    return ret
