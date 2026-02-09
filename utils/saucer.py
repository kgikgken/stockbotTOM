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
    close: np.ndarray,
    volume: np.ndarray | None = None,
    *,
    min_len: int,
    max_len: int,
    min_depth: float,
    max_depth: float,
    vertex_band: tuple[float, float] = (0.35, 0.65),
    min_progress: float = 0.95,
    depth_opt: float = 0.28,
) -> dict | None:
    """
    Returns dict:
      score (0..1), progress (0..), depth (0..1), length (bars), rim (price)

    Depth = (rim - bottom) / rim
    progress = current / rim

    Design goal (spec): "saucer" quality should NOT reward overly-deep bases.
    Guidance for classical "cup" depth is often cited around ~12%–33% (e.g., O'Neil/IBD),
    with deeper bases generally lower quality / higher risk. We enforce min/max depth and
    add a soft penalty away from depth_opt.
    """
    n = int(len(close))
    if n < min_len:
        return None
    if n > max_len:
        close = close[-max_len:]
        if volume is not None:
            volume = volume[-max_len:]
        n = int(len(close))

    if np.any(~np.isfinite(close)) or np.nanmin(close) <= 0:
        return None
    close = close.astype(float)

    # rims and bottom
    left_len = max(5, n // 4)
    right_len = max(5, n // 4)
    left_rim = float(np.max(close[:left_len]))
    right_rim = float(np.max(close[-right_len:]))
    rim = float(min(left_rim, right_rim))
    bottom_idx = int(np.argmin(close))
    bottom = float(close[bottom_idx])
    if rim <= 0 or bottom <= 0:
        return None

    depth = (rim - bottom) / rim
    if (depth < min_depth) or (depth > max_depth):
        return None

    vertex_pos = bottom_idx / max(1, n - 1)
    if not (vertex_band[0] <= vertex_pos <= vertex_band[1]):
        return None

    progress = float(close[-1] / rim)
    if progress < min_progress:
        return None

    # shape slopes
    if bottom_idx < 10 or (n - bottom_idx) < 10:
        return None
    x = np.arange(n, dtype=float)

    def _slope(xx: np.ndarray, yy: np.ndarray) -> float:
        xx = xx - xx.mean()
        denom = float(np.sum(xx * xx))
        if denom <= 0:
            return 0.0
        return float(np.sum(xx * (yy - yy.mean())) / denom)

    slope_l = _slope(x[: bottom_idx + 1], close[: bottom_idx + 1])
    slope_r = _slope(x[bottom_idx:], close[bottom_idx:])
    if slope_l >= 0:
        return None
    if slope_r <= 0:
        return None

    # reject sharp V-bottoms
    rets = np.diff(close) / close[:-1]
    if len(rets) >= 20:
        win = min(10, bottom_idx - 1, len(rets) - bottom_idx - 1) if bottom_idx > 1 else 0
        if win >= 3:
            local = float(np.mean(np.abs(rets[bottom_idx - win : bottom_idx + win])))
            overall = float(np.mean(np.abs(rets)))
            if overall > 0 and local > overall * 2.2:
                return None

    # volume confirmation (optional)
    vol_score = 0.0
    if volume is not None and len(volume) == len(close):
        v = volume.astype(float)
        if np.all(np.isfinite(v)) and float(np.nanmedian(v)) > 0:
            a = n // 3
            b = 2 * n // 3
            v1 = float(np.nanmedian(v[:a]))
            v2 = float(np.nanmedian(v[a:b]))
            v3 = float(np.nanmedian(v[b:]))
            if v1 > 0 and v2 > 0 and v3 > 0:
                dry = max(0.0, min(1.0, (v1 - v2) / v1))
                exp = max(0.0, min(1.0, (v3 - v2) / v3))
                vol_score = 0.5 * dry + 0.5 * exp

    # depth quality
    depth_pen = max(0.0, 1.0 - abs(depth - depth_opt) / max(1e-6, (max_depth - min_depth)))
    depth_pen = max(0.0, min(1.0, depth_pen))

    # progress quality
    prog_q = max(0.0, min(1.0, (min(progress, 1.05) - min_progress) / (1.05 - min_progress)))

    score = 0.45 * prog_q + 0.45 * depth_pen + 0.10 * vol_score

    return {
        "score": float(score),
        "progress": float(progress),
        "depth": float(depth),
        "length": int(n),
        "rim": float(rim),
    }
def scan_saucers(universe: list[dict], ohlc_map: dict[str, pd.DataFrame]) -> dict[str, list[SaucerHit]]:
    """
    Saucer (rounding bottom) scanner for D/W/M timeframes.

    Output dict keys: "D", "W", "M" -> list[SaucerHit] sorted by quality.
    Each timeframe is capped later in report layer (max 5 per TF).

    Notes:
    - We intentionally bias toward *moderate* depth bases. Extremely deep bases are filtered out.
    - Completion requires progress near the rim.
    - Minimal volume confirmation (dry-up then expansion) is used when volume exists.
    """

    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        # expects columns: Open/High/Low/Close/Volume
        if df is None or df.empty:
            return pd.DataFrame()
        x = df.copy()
        if not isinstance(x.index, pd.DatetimeIndex):
            x.index = pd.to_datetime(x.index, errors="coerce")
        x = x.dropna(subset=["Close"])
        if x.empty:
            return pd.DataFrame()
        ohlc = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        y = x.resample(rule).apply(ohlc).dropna(subset=["Close"])
        return y

    # Config tuned for "quality" bases.
    cfg = {
        "D": dict(lookback=220, min_len=70, max_len=220, min_depth=0.18, max_depth=0.45, min_progress=0.97, depth_opt=0.28, ma_win=50),
        "W": dict(lookback=160, min_len=26, max_len=120, min_depth=0.15, max_depth=0.40, min_progress=0.98, depth_opt=0.25, ma_win=30),
        "M": dict(lookback=80,  min_len=12, max_len=60,  min_depth=0.12, max_depth=0.33, min_progress=0.985, depth_opt=0.22, ma_win=10),
    }

    out: dict[str, list[SaucerHit]] = {"D": [], "W": [], "M": []}

    for row in universe:
        ticker = str(row.get("ticker") or row.get("code") or "").strip()
        if not ticker:
            continue
        df = ohlc_map.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        # Prepare D/W/M data
        df_d = df.copy()
        df_w = _resample(df, "W-FRI")
        df_m = _resample(df, "M")

        for tf, dfx in (("D", df_d), ("W", df_w), ("M", df_m)):
            c = cfg[tf]
            if dfx is None or dfx.empty or len(dfx) < c["min_len"]:
                continue

            dfx = dfx.tail(c["lookback"]).copy()

            close = dfx["Close"].to_numpy(dtype=float)
            volume = dfx["Volume"].to_numpy(dtype=float) if "Volume" in dfx.columns else None

            # MA sanity: last close should be above MA (helps avoid bases that are still broken)
            ma_win = int(c["ma_win"])
            if len(close) >= ma_win:
                ma = float(np.mean(close[-ma_win:]))
                if close[-1] < ma:
                    continue

            met = _saucer_score(
                close,
                volume,
                min_len=int(c["min_len"]),
                max_len=int(c["max_len"]),
                min_depth=float(c["min_depth"]),
                max_depth=float(c["max_depth"]),
                vertex_band=(0.35, 0.65),
                min_progress=float(c["min_progress"]),
                depth_opt=float(c["depth_opt"]),
            )
            if met is None:
                continue

            out[tf].append(
                SaucerHit(
                    ticker=ticker,
                    name=name,
                    sector=sector,
                    timeframe=tf,
                    rim_price=float(met["rim"]),
                    progress=float(met["progress"]),
                    depth=float(met["depth"]),
                    score=float(met["score"]),
                )
            )

    # Sort: score desc -> progress desc -> depth closer to optimal -> ticker
    def _sort_key(hit: SaucerHit):
        c = cfg[hit.timeframe]
        depth_opt = float(c["depth_opt"])
        return (
            hit.score,
            hit.progress,
            -abs(hit.depth - depth_opt),
            hit.rim_price,
            hit.ticker,
        )

    for tf in ("D", "W", "M"):
        out[tf].sort(key=_sort_key, reverse=True)

    return out
