from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import math

from utils.util import safe_float, adv20, atr_pct_last


# ---------------------------------------------------------------------
# Saucer / Cup-with-Handle scanner
# ---------------------------------------------------------------------
#
# Design intent (per your latest instructions):
# - Prefer LONG, smooth "cup/saucer" shapes.
# - Enter BEFORE a clean breakout (aggressive): inside the handle, close to the rim.
# - Reduce false-positives where a monotonic uptrend is misclassified as a cup.
#
# This implementation borrows the widely-used canonical constraints for
# "cup with handle" (moderate cup depth, handle in upper half, handle not too deep)
# and then tailors the entry zone to your aggressive "ココ" (pre-breakout) preference.
# ---------------------------------------------------------------------


@dataclass
class SaucerHit:
    ticker: str
    name: str
    sector: str
    tf: str  # 'D' | 'W' | 'M'

    rim_price: float        # resistance line (left rim reference)
    last_price: float       # last close in that timeframe
    atrp: float             # ATR% (timeframe)

    progress: float         # 0-1, 1 ~= reach rim
    depth: float            # 0-1, cup depth ratio
    cup_len: int            # bars from left rim to handle start (longer is better)

    # handle
    handle_low: float       # min close in handle segment
    handle_high: float      # max close in handle segment (near rim)
    handle_pb: float        # (rim-handle_low)/rim
    handle_len: int         # bars
    handle_vol_ratio: float # handle_vol / prev_vol (if available)

    # suggested execution levels (aggressive, pre-breakout)
    entry_low: float
    entry_high: float
    sl_price: float
    risk_pct: float

    score: float            # ranking score (internal)

    # Classification tier:
    # - "A": canonical cup-with-handle constraints (higher quality)
    # - "B": relaxed constraints used only when the market is sparse (keeps ideas flowing)
    tier: str = "A"


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


def _lin_slope(y: np.ndarray) -> float:
    """Return slope of y over x=[0..n-1] in 'y units per bar'."""
    y = np.asarray(y, dtype=float)
    if y.size < 3 or not np.isfinite(y).all():
        return float("nan")
    x = np.arange(y.size, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return float("nan")
    return float(((x * (y - y.mean())).sum()) / denom)



def _tick_size_jpx(price: float) -> float:
    """Approximate JPX (TSE) tick size table by price range.

    This is used to snap suggested levels so that rounding does not violate the risk cap.
    """
    try:
        p = float(price)
    except Exception:
        return 1.0
    if not np.isfinite(p) or p <= 0:
        return 1.0

    # Most common ranges for cash equities
    if p < 10_000:
        return 1.0
    if p < 30_000:
        return 5.0
    if p < 50_000:
        return 10.0
    if p < 100_000:
        return 10.0
    if p < 300_000:
        return 50.0
    if p < 500_000:
        return 100.0
    if p < 1_000_000:
        return 100.0
    if p < 3_000_000:
        return 500.0
    if p < 5_000_000:
        return 1_000.0
    if p < 10_000_000:
        return 1_000.0
    if p < 30_000_000:
        return 5_000.0
    return 10_000.0


def _floor_to_tick(price: float, tick: float) -> float:
    try:
        p = float(price)
        t = float(tick)
    except Exception:
        return float("nan")
    if not (np.isfinite(p) and np.isfinite(t) and t > 0):
        return float("nan")
    return float(math.floor(p / t) * t)


def _entry_zone_prebreak(
    rim: float,
    handle_pb: float,
    atrp: float,
    tf: str,
) -> Tuple[float, float]:
    """
    Aggressive pre-breakout entry zone close to rim, inside the handle.

    We intentionally *do not* use handle_low itself as the lower bound, because your
    "ココ" is in the upper part of the handle (before breakout), not at the deepest pullback.

    Zone widths adapt to:
    - handle depth (primary)
    - ATR% (secondary floor, to avoid ultra-tight zones that never fill)
    """
    if not (np.isfinite(rim) and rim > 0 and np.isfinite(handle_pb) and handle_pb > 0):
        return (float("nan"), float("nan"))

    pb_pct = float(handle_pb * 100.0)
    atrp = float(atrp) if np.isfinite(atrp) else 0.0

    # Caps by timeframe: we keep the zone "near rim" as you requested.
    low_cap = {"D": 2.5, "W": 4.0, "M": 6.0}.get(tf, 3.0)   # max distance from rim (lower bound)
    high_cap = {"D": 0.9, "W": 1.6, "M": 2.5}.get(tf, 1.2)  # max distance from rim (upper bound)

    # Upper bound: very close to rim.
    high_pct = max(0.15, 0.05 * atrp, 0.05 * pb_pct)  # %
    high_pct = float(np.clip(high_pct, 0.15, high_cap))

    # Lower bound: upper half of handle (around 0.35*pullback from rim), with ATR floor.
    low_pct = max(high_pct + 0.15, 0.12 * atrp, 0.35 * pb_pct)  # %
    low_pct = float(np.clip(low_pct, high_pct + 0.15, low_cap))

    entry_low = rim * (1.0 - low_pct / 100.0)
    entry_high = rim * (1.0 - high_pct / 100.0)

    if entry_high < entry_low:
        entry_low, entry_high = entry_high, entry_low

    return (float(entry_low), float(entry_high))


def _calc_sl_and_risk(
    *,
    entry_low: float,
    entry_high: float,
    handle_low_wick: float,
    last_price: float,
    atrp: float,
    tf: str,
) -> Tuple[float, float]:
    """Compute an SL just below handle low, and risk% at mid-entry."""
    if not (np.isfinite(entry_low) and np.isfinite(entry_high) and entry_low > 0 and entry_high > 0):
        return (float("nan"), float("nan"))

    entry_mid = (entry_low + entry_high) / 2.0

    # ATR in price units
    atrp = float(atrp) if np.isfinite(atrp) else 0.0
    ref_px = last_price if (np.isfinite(last_price) and last_price > 0) else entry_mid
    atr_px = ref_px * (atrp / 100.0) if (atrp > 0 and ref_px > 0) else 0.0

    # Pad below handle low: smaller on daily, larger on weekly/monthly.
    pad_atr = {"D": 0.55, "W": 0.65, "M": 1.00}.get(tf, 0.65)
    sl = handle_low_wick - pad_atr * atr_px if (np.isfinite(handle_low_wick) and handle_low_wick > 0) else float("nan")
    if not (np.isfinite(sl) and sl > 0):
        sl = handle_low_wick if (np.isfinite(handle_low_wick) and handle_low_wick > 0) else float("nan")

    # Risk%
    if np.isfinite(sl) and sl > 0 and entry_mid > 0:
        risk_pct = (entry_mid - sl) / entry_mid * 100.0
    else:
        risk_pct = float("nan")

    return (float(sl), float(risk_pct))


def _cup_with_handle_metrics(
    df: pd.DataFrame,
    *,
    tf: str,
    lookback: int,
    min_cup_len: int,
    min_progress: float,
    min_depth: float,
    max_depth: float,
    # left rim detection
    left_rim_frac: float,
    # rim symmetry
    rim_tol_below: float,
    rim_tol_above: float,
    # actionable pre-break constraint (how far the *last close* may be above rim)
    # Your "ココ" entry is pre-break, so this should be small.
    max_last_over_rim: float,
    # bottom shape
    min_bottom_pos: float,
    max_bottom_pos: float,
    min_bottom_width_frac: float,
    # right-side steepness (avoid J-curves / pure trends)
    max_rise_slope_ratio: float,
    # handle detection
    handle_win: int,
    handle_min_len: int,
    handle_max_len: int,
    handle_near_rim: float,
    handle_min_pb: float,
    handle_max_pb: float,
    handle_max_range: float,
    handle_min_upper_frac: float,
    handle_vol_ratio_max: float,
    # hard filter: if handle volume is materially higher than prior, drop (reduces false positives)
    handle_vol_ratio_hard_max: float,
    # risk control
    max_risk_pct: float = 8.0,
) -> Optional[Dict[str, float]]:
    """Return metrics dict or None."""
    if df is None or df.empty:
        return None
    if "Close" not in df.columns:
        return None

    seg = df.tail(int(lookback)).copy()
    seg = seg.dropna(subset=["Close"])
    if len(seg) < max(40, min_cup_len + 10):
        return None

    c = seg["Close"].astype(float)
    h = seg["High"].astype(float) if "High" in seg.columns else c
    l = seg["Low"].astype(float) if "Low" in seg.columns else c
    v = seg["Volume"].astype(float) if "Volume" in seg.columns else None

    n = int(len(seg))

    # ------------------
    # 1) Left rim (prior high) in early window
    # ------------------
    left_n = max(10, int(round(n * float(np.clip(left_rim_frac, 0.15, 0.60)))))
    left_seg = c.iloc[:left_n]
    rim = float(left_seg.max())
    if not np.isfinite(rim) or rim <= 0:
        return None
    idx_left = int(np.nanargmax(left_seg.values))

    # ------------------
    # 2) Bottom after left rim
    # ------------------
    after = c.iloc[idx_left:]
    if len(after) < 20:
        return None
    bottom = float(after.min())
    if not np.isfinite(bottom) or bottom <= 0:
        return None
    idx_bottom = idx_left + int(np.nanargmin(after.values))

    # Require some descent duration
    if (idx_bottom - idx_left) < int(max(6, round(n * 0.05))):
        return None

    # ------------------
    # 3) Handle detection (must be near end for D/W)
    # ------------------
    if handle_win <= 0 or n < (handle_win + 10):
        return None

    hs = c.iloc[-int(handle_win):]
    idx_peak_in_hs = int(np.nanargmax(hs.values))
    idx_handle_start = (n - int(handle_win)) + idx_peak_in_hs

    # handle segment from peak to end
    handle_seg_close = c.iloc[idx_handle_start:]
    handle_seg_high = h.iloc[idx_handle_start:]
    handle_seg_low = l.iloc[idx_handle_start:]
    handle_len = int(len(handle_seg_close))

    if handle_len < int(handle_min_len) or handle_len > int(handle_max_len):
        return None

    handle_high_close = float(handle_seg_close.max())
    handle_low_close = float(handle_seg_close.min())
    handle_low_wick = float(handle_seg_low.min()) if len(handle_seg_low) else handle_low_close

    if not (np.isfinite(handle_high_close) and np.isfinite(handle_low_close)):
        return None

    # handle must be near rim (touch/approach)
    if handle_high_close < rim * (1.0 - handle_near_rim):
        return None

    handle_pb = float((rim - handle_low_close) / rim) if rim > 0 else float("nan")
    if not (np.isfinite(handle_pb) and handle_min_pb <= handle_pb <= handle_max_pb):
        return None

    handle_range = float((handle_high_close - handle_low_close) / rim) if rim > 0 else float("nan")
    if not (np.isfinite(handle_range) and handle_range <= handle_max_range):
        return None

    # handle should stay in upper part of cup (O'Neil style)
    cup_mid = bottom + handle_min_upper_frac * (rim - bottom)
    if handle_low_close < cup_mid:
        return None

    last = float(c.iloc[-1])
    if not (np.isfinite(last) and last > 0):
        return None

    # Too extended above rim => already broke out (not pre-breakout).
    # NOTE: this is intentionally much stricter than rim symmetry tolerances.
    if last > rim * (1.0 + float(max_last_over_rim)):
        return None

    # ------------------
    # 4) Cup length is left rim -> handle start
    # ------------------
    cup_len = int(idx_handle_start - idx_left + 1)
    if cup_len < int(min_cup_len):
        return None

    # ------------------
    # 5) Right rim symmetry check (avoid monotonic trends)
    # ------------------
    right_seg = c.iloc[max(idx_bottom, idx_left):idx_handle_start + 1]
    right_rim = float(right_seg.max()) if len(right_seg) else float("nan")
    if not (np.isfinite(right_rim) and right_rim > 0):
        return None
    if right_rim < rim * (1.0 - rim_tol_below):
        return None
    if right_rim > rim * (1.0 + rim_tol_above):
        return None

    # ------------------
    # 6) Cup depth and completion
    # ------------------
    depth = float((rim - bottom) / rim)
    if not (np.isfinite(depth) and min_depth <= depth <= max_depth):
        return None

    denom = max(1e-9, (rim - bottom))
    progress = float((last - bottom) / denom)
    if not (np.isfinite(progress) and progress >= min_progress):
        return None

    # Bottom position should be around the middle of the cup
    rel_bottom = (idx_bottom - idx_left) / max(1e-9, float(cup_len))
    if not (min_bottom_pos <= rel_bottom <= max_bottom_pos):
        return None

    # Bottom width check (avoid sharp V bottoms)
    cup_seg = c.iloc[idx_left:idx_handle_start + 1]
    cup_ma = cup_seg.rolling(7, min_periods=4).mean()
    band = bottom + 0.18 * (rim - bottom)
    bottom_width = int(np.nansum((cup_ma <= band).values))
    if bottom_width < int(max(3, round(cup_len * float(min_bottom_width_frac)))):
        return None

    # Right-side steepness check (avoid J-curves / pure trend)
    left_dur = max(1, idx_bottom - idx_left)
    right_dur = max(1, idx_handle_start - idx_bottom)
    left_slope = (rim - bottom) / float(left_dur)
    right_slope = (right_rim - bottom) / float(right_dur)
    if left_slope > 0 and right_slope / left_slope > max_rise_slope_ratio:
        return None

    # Handle slope should be flat-to-down (avoid runaway into breakout)
    h_slope = _lin_slope(handle_seg_close.values)
    if np.isfinite(h_slope) and h_slope > (rim * 0.0015):  # per-bar rise too large
        return None

    # Volume dry-up in handle (optional but valuable)
    handle_vol_ratio = float("nan")
    vol_bonus = 0.0
    if v is not None and (v.notna().sum() >= (handle_len + 5)):
        hv = float(v.iloc[idx_handle_start:].mean())
        pv_start = max(0, idx_handle_start - handle_len)
        pv = float(v.iloc[pv_start:idx_handle_start].mean()) if idx_handle_start > pv_start else float("nan")
        if np.isfinite(hv) and np.isfinite(pv) and pv > 0:
            handle_vol_ratio = hv / pv
            # Hard filter: handle should *not* show a large volume expansion versus prior.
            # (Dry-up is ideal; flat is acceptable; expansion tends to be noisy/late.)
            if np.isfinite(handle_vol_ratio) and handle_vol_ratio_hard_max > 0 and handle_vol_ratio > handle_vol_ratio_hard_max:
                return None
            if handle_vol_ratio <= handle_vol_ratio_max:
                vol_bonus = 0.25
            else:
                vol_bonus = -0.15

    # Entry zone + SL + risk
    atrp = atr_pct_last(seg)
    entry_low, entry_high = _entry_zone_prebreak(rim, handle_pb, atrp, tf)
    sl_price, risk_pct = _calc_sl_and_risk(
        entry_low=entry_low,
        entry_high=entry_high,
        handle_low_wick=handle_low_wick,
        last_price=last,
        atrp=atrp,
        tf=tf,
    )

    # Enforce the max risk *across the actionable entry zone*.
    # We keep your preferred near-rim zone shape, but if the upper end would exceed
    # the risk cap, we cap the entry_high down to the highest price that still
    # satisfies max_risk_pct given the computed SL.
    if (
        np.isfinite(entry_low)
        and np.isfinite(entry_high)
        and entry_low > 0
        and entry_high > 0
        and np.isfinite(sl_price)
        and sl_price > 0
        and max_risk_pct > 0
    ):
        # highest entry that keeps risk <= cap
        cap_high = sl_price / (1.0 - max_risk_pct / 100.0)
        if np.isfinite(cap_high) and cap_high > 0:
            entry_high = float(min(entry_high, cap_high))
            if entry_high < entry_low:
                return None
            # recompute mid-zone risk after capping
            entry_mid = (entry_low + entry_high) / 2.0
            risk_pct = (entry_mid - sl_price) / entry_mid * 100.0 if entry_mid > 0 else float("nan")

    
    # Snap levels to tradable ticks (conservative) so that report rounding does not
    # accidentally push the effective risk above the cap (e.g. 8.01% due to rounding).
    tick = _tick_size_jpx(rim)
    if np.isfinite(tick) and tick > 0:
        entry_low = _floor_to_tick(entry_low, tick)
        entry_high = _floor_to_tick(entry_high, tick)
        sl_price = _floor_to_tick(sl_price, tick)

        if not (
            np.isfinite(entry_low)
            and np.isfinite(entry_high)
            and np.isfinite(sl_price)
            and entry_low > 0
            and entry_high > 0
            and sl_price > 0
        ):
            return None

        # Re-apply strict cap on the snapped values (ensures max risk holds on real order prices).
        cap_high = sl_price / (1.0 - max_risk_pct / 100.0)
        cap_high = _floor_to_tick(cap_high, tick)
        if np.isfinite(cap_high) and cap_high > 0:
            entry_high = float(min(entry_high, cap_high))

        if entry_high < entry_low:
            return None

        entry_mid = (entry_low + entry_high) / 2.0
        risk_pct = (entry_mid - sl_price) / entry_mid * 100.0 if entry_mid > 0 else float("nan")

    if np.isfinite(risk_pct) and risk_pct > max_risk_pct:
        return None

    # Roughness (smooth bowls preferred)
    rough = float(np.nanstd((cup_seg - cup_ma) / rim))

    # Penalties around ideal depth and handle depth (canonical ranges)
    depth_ideal = 0.28 if tf in ("D", "W") else 0.35
    depth_pen = abs(depth - depth_ideal) / max(1e-6, (max_depth - min_depth))

    handle_ideal = 0.08 if tf == "D" else (0.10 if tf == "W" else 0.12)
    handle_pen = abs(handle_pb - handle_ideal) / max(1e-6, (handle_max_pb - handle_min_pb))

    # Length bonus: strong (your preference).
    # Use log scaling so "longer is better" but doesn't explode.
    length_bonus = float(np.log1p(cup_len) / np.log1p(lookback) * 1.35)

    # Rim mismatch penalty
    rim_gap = abs(right_rim - rim) / rim
    rim_pen = float(np.clip(rim_gap / max(1e-9, rim_tol_below), 0.0, 2.0))

    # Final score
    score = (
        (min(1.20, progress) * 2.20)
        + length_bonus
        - (depth_pen * 0.80)
        - (handle_pen * 0.60)
        - (rough * 8.0)
        - (rim_pen * 0.35)
        + vol_bonus
    )

    return {
        "rim": float(rim),
        "right_rim": float(right_rim),
        "bottom": float(bottom),
        "last": float(last),
        "depth": float(depth),
        "progress": float(progress),
        "cup_len": int(cup_len),
        "handle_low": float(handle_low_close),
        "handle_high": float(handle_high_close),
        "handle_pb": float(handle_pb),
        "handle_len": int(handle_len),
        "handle_vol_ratio": float(handle_vol_ratio) if np.isfinite(handle_vol_ratio) else float("nan"),
        "atrp": float(atrp) if np.isfinite(atrp) else float("nan"),
        "entry_low": float(entry_low),
        "entry_high": float(entry_high),
        "sl": float(sl_price) if np.isfinite(sl_price) else float("nan"),
        "risk_pct": float(risk_pct) if np.isfinite(risk_pct) else float("nan"),
        "score": float(score),
    }


def scan_saucers(
    ohlc_map: Dict[str, pd.DataFrame],
    uni: pd.DataFrame,
    tcol: str,
    *,
    max_each: int = 5,
) -> Dict[str, List[Dict]]:
    """Scan saucers / cups on D/W/M.

    Notes:
    - D/W require a handle and produce an aggressive pre-breakout entry zone near rim.
    - Monthly is optional and more permissive (handles are often unclear on monthly bars).
    """
    if uni is None or uni.empty:
        return {"D": [], "W": [], "M": []}

    out: Dict[str, List[SaucerHit]] = {"D": [], "W": [], "M": []}

    for _, row in uni.iterrows():
        ticker = str(row.get(tcol, "")).strip()
        if not ticker:
            continue

        df = ohlc_map.get(ticker)
        if df is None or df.empty or len(df) < 220:
            continue

        # Sanity / liquidity guardrails
        last_close = safe_float(df["Close"].iloc[-1], np.nan)
        if not np.isfinite(last_close):
            continue
        if last_close < 50.0 or last_close > 200000.0:
            continue

        adv = adv20(df)
        atrp_d = atr_pct_last(df)
        if (not np.isfinite(adv)) or adv < 120e6:
            continue
        if (not np.isfinite(atrp_d)) or atrp_d < 0.6:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        # ------------------
        # Daily (Cup-with-handle, practical mode)
        # ------------------
        # v13 was deliberately strict; in practice it can produce "no hits" on many days.
        # Here we relax *only* the parts that commonly suppress good-looking handles:
        # - allow slightly shallower / less-complete cups
        # - allow the left rim to occur a bit later in the early window
        # - widen rim symmetry tolerances a touch
        # - permit small rim overshoots while still staying "pre-breakout"
        # - tolerate handle volume being flat-ish (dry-up is still preferred)
        met = _cup_with_handle_metrics(
            df,
            tf="D",
            lookback=420,
            min_cup_len=90,
            min_progress=0.84,
            min_depth=0.10,
            max_depth=0.55,
            left_rim_frac=0.35,
            rim_tol_below=0.10,
            rim_tol_above=0.05,
            max_last_over_rim=0.007,
            min_bottom_pos=0.28,
            max_bottom_pos=0.72,
            min_bottom_width_frac=0.03,
            max_rise_slope_ratio=2.80,
            handle_win=30,
            handle_min_len=5,
            handle_max_len=30,
            handle_near_rim=0.045,
            handle_min_pb=0.02,
            handle_max_pb=0.18,
            handle_max_range=0.14,
            handle_min_upper_frac=0.45,
            handle_vol_ratio_max=1.00,
            handle_vol_ratio_hard_max=1.45,
            max_risk_pct=8.0,
        )
        tier = "A" if met else ""

        # Fallback (tier B): only slightly looser; keeps the list from going empty
        # while still enforcing: long cup + handle + pre-breakout proximity + risk cap.
        if met is None:
            met = _cup_with_handle_metrics(
                df,
                tf="D",
                lookback=420,
                min_cup_len=80,
                min_progress=0.80,
                min_depth=0.08,
                max_depth=0.55,
                left_rim_frac=0.40,
                rim_tol_below=0.12,
                rim_tol_above=0.07,
                max_last_over_rim=0.007,
                min_bottom_pos=0.26,
                max_bottom_pos=0.74,
                min_bottom_width_frac=0.02,
                max_rise_slope_ratio=3.20,
                handle_win=35,
                handle_min_len=5,
                handle_max_len=35,
                handle_near_rim=0.06,
                handle_min_pb=0.02,
                handle_max_pb=0.22,
                handle_max_range=0.18,
                handle_min_upper_frac=0.40,
                handle_vol_ratio_max=1.05,
                handle_vol_ratio_hard_max=1.45,
                max_risk_pct=8.0,
            )
            tier = "B" if met else ""

        if met:
            out["D"].append(
                SaucerHit(
                    ticker=ticker,
                    name=name,
                    sector=sector,
                    tf="D",
                    rim_price=float(met["rim"]),
                    last_price=float(met["last"]),
                    atrp=float(met.get("atrp") or atrp_d),
                    progress=float(met["progress"]),
                    depth=float(met["depth"]),
                    cup_len=int(met["cup_len"]),
                    handle_low=float(met["handle_low"]),
                    handle_high=float(met["handle_high"]),
                    handle_pb=float(met["handle_pb"]),
                    handle_len=int(met.get("handle_len", 0) or 0),
                    handle_vol_ratio=float(met.get("handle_vol_ratio", float("nan"))),
                    entry_low=float(met.get("entry_low", float("nan"))),
                    entry_high=float(met.get("entry_high", float("nan"))),
                    sl_price=float(met.get("sl", float("nan"))),
                    risk_pct=float(met.get("risk_pct", float("nan"))),
                    score=float(met["score"]),
                    tier=tier or "A",
                )
            )

        # ------------------
        # Weekly (practical mode)
        # ------------------
        try:
            w = _resample(df, "W-FRI")
        except Exception:
            w = None
        if w is not None and not w.empty and len(w) >= 70:
            met = _cup_with_handle_metrics(
                w,
                tf="W",
                lookback=182,
                min_cup_len=32,
                min_progress=0.82,
                min_depth=0.10,
                max_depth=0.60,
                left_rim_frac=0.35,
                rim_tol_below=0.12,
                rim_tol_above=0.07,
                max_last_over_rim=0.010,
                min_bottom_pos=0.26,
                max_bottom_pos=0.74,
                min_bottom_width_frac=0.02,
                max_rise_slope_ratio=3.00,
                handle_win=14,
                handle_min_len=3,
                handle_max_len=14,
                handle_near_rim=0.06,
                handle_min_pb=0.03,
                handle_max_pb=0.22,
                handle_max_range=0.20,
                handle_min_upper_frac=0.40,
                handle_vol_ratio_max=1.05,
                handle_vol_ratio_hard_max=1.45,
                max_risk_pct=8.0,
            )
            tier_w = "A" if met else ""

            # Weekly fallback (tier B)
            if met is None:
                met = _cup_with_handle_metrics(
                    w,
                    tf="W",
                    lookback=182,
                    min_cup_len=28,
                    min_progress=0.78,
                    min_depth=0.08,
                    max_depth=0.62,
                    left_rim_frac=0.40,
                    rim_tol_below=0.14,
                    rim_tol_above=0.09,
                    max_last_over_rim=0.014,
                    min_bottom_pos=0.24,
                    max_bottom_pos=0.76,
                    min_bottom_width_frac=0.015,
                    max_rise_slope_ratio=3.40,
                    handle_win=16,
                    handle_min_len=3,
                    handle_max_len=16,
                    handle_near_rim=0.07,
                    handle_min_pb=0.03,
                    handle_max_pb=0.25,
                    handle_max_range=0.24,
                    handle_min_upper_frac=0.38,
                    handle_vol_ratio_max=1.10,
                    handle_vol_ratio_hard_max=1.55,
                    max_risk_pct=8.0,
                )
                tier_w = "B" if met else ""

            if met:
                atrp_w = atr_pct_last(w)
                out["W"].append(
                    SaucerHit(
                        ticker=ticker,
                        name=name,
                        sector=sector,
                        tf="W",
                        rim_price=float(met["rim"]),
                        last_price=float(met["last"]),
                        atrp=float(met.get("atrp") or atrp_w),
                        progress=float(met["progress"]),
                        depth=float(met["depth"]),
                        cup_len=int(met["cup_len"]),
                        handle_low=float(met["handle_low"]),
                        handle_high=float(met["handle_high"]),
                        handle_pb=float(met["handle_pb"]),
                        handle_len=int(met.get("handle_len", 0) or 0),
                        handle_vol_ratio=float(met.get("handle_vol_ratio", float("nan"))),
                        entry_low=float(met.get("entry_low", float("nan"))),
                        entry_high=float(met.get("entry_high", float("nan"))),
                        sl_price=float(met.get("sl", float("nan"))),
                        risk_pct=float(met.get("risk_pct", float("nan"))),
                        score=float(met["score"]),
                        tier=tier_w or "A",
                    )
                )

        # ------------------
        # Monthly (more permissive)
        # ------------------
        try:
            m = _resample(df, "ME")
        except Exception:
            m = None
        if m is not None and not m.empty and len(m) >= 40:
            # Monthly "cup" is often multi-year; keep depth a bit wider.
            # We still use the same engine but relax handle constraints by giving wide handle limits
            # and a larger handle window.
            met = _cup_with_handle_metrics(
                m,
                tf="M",
                lookback=120,
                min_cup_len=24,
                min_progress=0.82,
                min_depth=0.12,
                max_depth=0.70,
                left_rim_frac=0.22,
                rim_tol_below=0.14,
                rim_tol_above=0.08,
                max_last_over_rim=0.010,
                min_bottom_pos=0.22,
                max_bottom_pos=0.78,
                min_bottom_width_frac=0.02,
                max_rise_slope_ratio=2.80,
                handle_win=10,
                handle_min_len=2,
                handle_max_len=18,
                handle_near_rim=0.06,
                handle_min_pb=0.02,
                handle_max_pb=0.30,
                handle_max_range=0.30,
                handle_min_upper_frac=0.40,
                handle_vol_ratio_max=1.10,
                handle_vol_ratio_hard_max=1.45,
                max_risk_pct=12.0,
            )
            if met:
                atrp_m = atr_pct_last(m)
                out["M"].append(
                    SaucerHit(
                        ticker=ticker,
                        name=name,
                        sector=sector,
                        tf="M",
                        rim_price=float(met["rim"]),
                        last_price=float(met["last"]),
                        atrp=float(met.get("atrp") or atrp_m),
                        progress=float(met["progress"]),
                        depth=float(met["depth"]),
                        cup_len=int(met["cup_len"]),
                        handle_low=float(met["handle_low"]),
                        handle_high=float(met["handle_high"]),
                        handle_pb=float(met["handle_pb"]),
                        handle_len=int(met.get("handle_len", 0) or 0),
                        handle_vol_ratio=float(met.get("handle_vol_ratio", float("nan"))),
                        entry_low=float(met.get("entry_low", float("nan"))),
                        entry_high=float(met.get("entry_high", float("nan"))),
                        sl_price=float(met.get("sl", float("nan"))),
                        risk_pct=float(met.get("risk_pct", float("nan"))),
                        score=float(met["score"]),
                    )
                )

    # Sort and truncate
    ret: Dict[str, List[Dict]] = {}
    for tf in ("D", "W", "M"):
        hits = out[tf]
        hits.sort(
            key=lambda x: (
                1 if str(getattr(x, "tier", "A")) == "A" else 0,
                x.score,
                x.cup_len,
                x.progress,
                -x.risk_pct if np.isfinite(x.risk_pct) else 0.0,
                x.ticker,
            ),
            reverse=True,
        )
        hits = hits[: max_each]
        ret[tf] = [
            {
                "ticker": h.ticker,
                "name": h.name,
                "sector": h.sector,
                "tf": h.tf,
                "tier": getattr(h, "tier", "A"),
                "rim": float(h.rim_price),
                "last": float(h.last_price),
                "atrp": float(h.atrp),
                "progress": float(h.progress),
                "depth": float(h.depth),
                "cup_len": int(h.cup_len),
                "handle_low": float(h.handle_low),
                "handle_high": float(h.handle_high),
                "handle_pb": float(h.handle_pb),
                "handle_len": int(h.handle_len),
                "handle_vol_ratio": float(h.handle_vol_ratio) if np.isfinite(h.handle_vol_ratio) else None,
                "entry_low": float(h.entry_low) if np.isfinite(h.entry_low) else None,
                "entry_high": float(h.entry_high) if np.isfinite(h.entry_high) else None,
                "sl": float(h.sl_price) if np.isfinite(h.sl_price) else None,
                "risk_pct": float(h.risk_pct) if np.isfinite(h.risk_pct) else None,
                "score": float(h.score),
            }
            for h in hits
        ]

    return ret