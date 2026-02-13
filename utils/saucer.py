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

    rim_price: float        # "left rim" target (prior high)
    last_price: float       # last close in that timeframe
    atrp: float             # ATR% (timeframe)

    progress: float         # 0-1, 1 ~= reach rim
    depth: float            # 0-1, cup depth ratio
    cup_len: int            # bars from left rim to current (longer is better)

    handle_low: float       # min close in handle window
    handle_high: float      # max close in handle window
    handle_pb: float        # (rim-handle_low)/rim
    handle_ok: bool

    score: float            # ranking score (internal)


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
    # shape controls
    left_rim_frac: float = 0.35,
    min_bottom_pos: float = 0.25,
    max_bottom_pos: float = 0.70,
    min_descent_frac: float = 0.08,
    # handle controls (your "ココ" entry)
    handle_win: int = 20,
    handle_min_pb: float = 0.006,
    handle_max_pb: float = 0.08,
    handle_near_rim: float = 0.012,
    handle_max_range: float = 0.10,
    require_handle: bool = True,
) -> Optional[Dict[str, float]]:
    """Return saucer metrics or None.

    IMPORTANT: rim is defined as a *left rim* (prior high) inside the early part
    of the lookback window. This avoids mistakenly treating monotonic uptrends
    as "saucers" (where the max is at the end).

    Your intent: enter *before* a clean breakout, around a small dip/handle near the rim.
    Therefore we optionally require a handle-like pullback near the rim.
    """

    c = close.astype(float).dropna()
    if len(c) < max(min_len, lookback + 5):
        return None

    seg = c.tail(lookback)
    n = int(len(seg))
    if n < max(30, min_len):
        return None

    # --- Define LEFT rim (prior high) within early window.
    left_n = max(10, int(round(n * float(np.clip(left_rim_frac, 0.15, 0.60)))))
    left_seg = seg.iloc[:left_n]
    rim = float(left_seg.max())
    if not np.isfinite(rim) or rim <= 0:
        return None
    idx_left_rim = int(np.nanargmax(left_seg.values))

    # --- Define bottom AFTER left rim.
    after_rim = seg.iloc[idx_left_rim:]
    if len(after_rim) < 10:
        return None
    bottom = float(after_rim.min())
    if not np.isfinite(bottom) or bottom <= 0:
        return None
    idx_bottom = idx_left_rim + int(np.nanargmin(after_rim.values))

    # Require meaningful descent portion.
    if (idx_bottom - idx_left_rim) < int(round(n * float(np.clip(min_descent_frac, 0.02, 0.25)))):
        return None

    # Bottom should be "around the middle" (avoid trend up / snapback V).
    lo_b = int(round(n * float(np.clip(min_bottom_pos, 0.05, 0.45))))
    hi_b = int(round(n * float(np.clip(max_bottom_pos, 0.55, 0.90))))
    if not (lo_b <= idx_bottom <= hi_b):
        return None

    last = float(seg.iloc[-1])
    if not np.isfinite(last) or last <= 0:
        return None

    # Depth based on left rim.
    depth = (rim - bottom) / rim
    if not (np.isfinite(depth) and min_depth <= depth <= max_depth):
        return None

    # Progress: how close last is to LEFT rim.
    denom = max(1e-9, (rim - bottom))
    progress = (last - bottom) / denom
    if not (np.isfinite(progress) and progress >= min_progress):
        return None

    # Too extended above rim => already broke out / not "before breakout".
    if last > rim * (1.0 + vertex_band):
        return None

    # Cup length: bars from left rim to end.
    cup_len = int(n - idx_left_rim)
    if cup_len < int(min_len):
        return None

    # --- Handle detection (your "ココ" dip near rim).
    handle_low = float("nan")
    handle_high = float("nan")
    handle_pb = float("nan")
    handle_ok = False

    if handle_win and n >= max(5, int(handle_win)):
        hs = seg.tail(int(handle_win))
        handle_low = float(hs.min())
        handle_high = float(hs.max())
        if np.isfinite(handle_low) and np.isfinite(handle_high) and rim > 0:
            handle_pb = float((rim - handle_low) / rim)  # pullback from rim
            handle_range = float((handle_high - handle_low) / rim)
            # Conditions:
            # 1) handle touches near the rim (recent highs close to rim)
            # 2) pullback depth within a reasonable handle range
            # 3) handle range not too wide (avoid chaotic chop)
            c1 = handle_high >= rim * (1.0 - handle_near_rim)
            c2 = (handle_min_pb <= handle_pb <= handle_max_pb)
            c3 = (handle_range <= handle_max_range)
            handle_ok = bool(c1 and c2 and c3)

    if require_handle and (not handle_ok):
        return None

    # Roughness penalty: smoother bowls are preferred.
    ma = seg.rolling(5, min_periods=3).mean()
    rough = float(np.nanstd((seg - ma) / rim))

    # Depth preference: closer to target (center of range) is better.
    target = (min_depth + max_depth) / 2.0
    width = max(1e-6, (max_depth - min_depth))
    depth_pen = abs(depth - target) / width

    # Length bonus: longer is better. Stronger weight per your preference.
    length_ratio = (cup_len - float(min_len)) / max(1e-9, float(lookback - min_len))
    length_ratio = float(np.clip(length_ratio, 0.0, 1.0))
    length_bonus = float(np.sqrt(length_ratio) * 1.15)

    handle_bonus = 0.20 if handle_ok else -0.20

    # Score: completion first, then depth, smoothness, length, and handle quality.
    score = (progress * 2.0) - (depth_pen * 0.60) - (rough * 8.0) + length_bonus + handle_bonus

    return {
        "rim": rim,
        "bottom": bottom,
        "last": last,
        "depth": float(depth),
        "progress": float(progress),
        "cup_len": int(cup_len),
        "handle_low": float(handle_low) if np.isfinite(handle_low) else float("nan"),
        "handle_high": float(handle_high) if np.isfinite(handle_high) else float("nan"),
        "handle_pb": float(handle_pb) if np.isfinite(handle_pb) else float("nan"),
        "handle_ok": 1.0 if handle_ok else 0.0,
        "score": float(score),
    }


def scan_saucers(
    ohlc_map: Dict[str, pd.DataFrame],
    uni: pd.DataFrame,
    tcol: str,
    *,
    max_each: int = 5,
) -> Dict[str, List[Dict]]:
    """Scan saucers on D/W/M.

    - Uses *left rim* (prior high) definition to avoid false positives on monotonic uptrends.
    - Prefers longer cups.
    - Requires a handle-like pullback near rim for D/W (your entry intent).
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

        # Sanity / liquidity guardrails
        last_close = safe_float(df["Close"].iloc[-1], np.nan)
        if not np.isfinite(last_close):
            continue
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

        # Daily
        met = _saucer_score(
            df["Close"],
            min_len=160,
            lookback=260,
            min_progress=0.92,
            min_depth=0.35,
            max_depth=0.75,
            vertex_band=0.03,
            left_rim_frac=0.35,
            min_bottom_pos=0.25,
            max_bottom_pos=0.70,
            min_descent_frac=0.08,
            handle_win=20,
            handle_min_pb=0.006,
            handle_max_pb=0.08,
            handle_near_rim=0.012,
            handle_max_range=0.10,
            require_handle=True,
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
                    cup_len=int(met.get("cup_len", 0) or 0),
                    handle_low=float(met.get("handle_low", np.nan)),
                    handle_high=float(met.get("handle_high", np.nan)),
                    handle_pb=float(met.get("handle_pb", np.nan)),
                    handle_ok=bool(float(met.get("handle_ok", 0.0)) > 0.5),
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
                lookback=104,
                min_progress=0.92,
                min_depth=0.45,
                max_depth=0.75,
                vertex_band=0.05,
                left_rim_frac=0.35,
                min_bottom_pos=0.25,
                max_bottom_pos=0.72,
                min_descent_frac=0.08,
                handle_win=10,
                handle_min_pb=0.01,
                handle_max_pb=0.12,
                handle_near_rim=0.015,
                handle_max_range=0.14,
                require_handle=True,
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
                        cup_len=int(met.get("cup_len", 0) or 0),
                        handle_low=float(met.get("handle_low", np.nan)),
                        handle_high=float(met.get("handle_high", np.nan)),
                        handle_pb=float(met.get("handle_pb", np.nan)),
                        handle_ok=bool(float(met.get("handle_ok", 0.0)) > 0.5),
                        score=float(met["score"]),
                    )
                )

        # Monthly (handle requirement disabled; monthly handles are not always clean)
        try:
            m = _resample(df, "ME")
        except Exception:
            m = None
        if m is not None and not m.empty and len(m) >= 36:
            atrp_m = atr_pct_last(m)
            met = _saucer_score(
                m["Close"],
                min_len=30,
                lookback=120,
                min_progress=0.90,
                min_depth=0.55,
                max_depth=0.80,
                vertex_band=0.08,
                left_rim_frac=0.35,
                min_bottom_pos=0.22,
                max_bottom_pos=0.75,
                min_descent_frac=0.06,
                handle_win=6,
                handle_min_pb=0.02,
                handle_max_pb=0.18,
                handle_near_rim=0.03,
                handle_max_range=0.22,
                require_handle=False,
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
                        cup_len=int(met.get("cup_len", 0) or 0),
                        handle_low=float(met.get("handle_low", np.nan)),
                        handle_high=float(met.get("handle_high", np.nan)),
                        handle_pb=float(met.get("handle_pb", np.nan)),
                        handle_ok=bool(float(met.get("handle_ok", 0.0)) > 0.5),
                        score=float(met["score"]),
                    )
                )

    # Sort and truncate
    ret: Dict[str, List[Dict]] = {}
    for tf in ("D", "W", "M"):
        hits = out[tf]
        hits.sort(
            key=lambda x: (x.score, x.cup_len, x.progress, x.depth, x.rim_price, x.ticker),
            reverse=True,
        )
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
                "cup_len": int(h.cup_len),
                "handle_low": float(h.handle_low) if np.isfinite(h.handle_low) else None,
                "handle_high": float(h.handle_high) if np.isfinite(h.handle_high) else None,
                "handle_pb": float(h.handle_pb) if np.isfinite(h.handle_pb) else None,
                "handle_ok": bool(h.handle_ok),
                "score": float(h.score),
            }
            for h in hits
        ]

    return ret
