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
        # limit per timeframe
    try:
        me = int(max_each)
    except Exception:
        me = 5
    if me > 0:
        for tf in list(out.keys()):
            out[tf] = out[tf][:me]

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
    rim_frac: float = 1.0,
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
def scan_saucers(ohlc_map, universe, tcol: str | None = None, max_each: int = 5):
    """Scan saucer-bottom candidates on D/W/M timeframes.

    Parameters
    ----------
    ohlc_map : dict[str, pandas.DataFrame]
        Per-ticker daily OHLCV (index: datetime-like).
    universe : pandas.DataFrame | list | tuple | set | dict
        Universe source. If DataFrame, will auto-detect ticker/name/sector columns.
        If list/set/tuple, elements may be ticker strings or dict-like rows.
    tcol : str | None
        Optional ticker column name when universe is a DataFrame.
    max_each : int
        Max tickers per timeframe to return.

    Returns
    -------
    dict[str, list[dict]]
        Keys: 'D','W','M' (daily/weekly/monthly). Values: list of dicts with
        ticker/name/sector/timeframe/entry_rim/progress/depth.
    """
    # ---- normalize universe -> iterable of (ticker, name, sector) ----
    ticker_meta = {}

    def _coerce_meta_from_row(row):
        if isinstance(row, dict):
            t = str(row.get("ticker") or row.get("code") or row.get("symbol") or "").strip()
            if not t:
                return None
            name = row.get("name") or row.get("company") or row.get("銘柄名") or ""
            sector = row.get("sector") or row.get("industry") or row.get("業種") or ""
            return t, str(name) if name is not None else "", str(sector) if sector is not None else ""
        # treat as ticker string
        t = str(row).strip()
        if not t:
            return None
        return t, "", ""

    if isinstance(universe, pd.DataFrame):
        cols = list(universe.columns)
        if tcol is None:
            # auto-detect ticker column
            for cand in ("ticker", "code", "symbol", "銘柄コード"):
                for c in cols:
                    if str(c).lower() == cand:
                        tcol = c
                        break
                if tcol is not None:
                    break
            if tcol is None and len(cols) > 0:
                # fallback: first column
                tcol = cols[0]

        name_col = None
        for cand in ("name", "company", "銘柄名"):
            for c in cols:
                if str(c).lower() == cand:
                    name_col = c
                    break
            if name_col is not None:
                break

        sector_col = None
        for cand in ("sector", "industry", "業種"):
            for c in cols:
                if str(c).lower() == cand:
                    sector_col = c
                    break
            if sector_col is not None:
                break

        for _, r in universe.iterrows():
            t = str(r.get(tcol, "")).strip()
            if not t:
                continue
            name = str(r.get(name_col, "")).strip() if name_col else ""
            sector = str(r.get(sector_col, "")).strip() if sector_col else ""
            ticker_meta[t] = (name, sector)

        tickers = list(ticker_meta.keys())

    elif isinstance(universe, (list, tuple, set)):
        tickers = []
        for row in universe:
            meta = _coerce_meta_from_row(row)
            if not meta:
                continue
            t, name, sector = meta
            tickers.append(t)
            if t not in ticker_meta:
                ticker_meta[t] = (name, sector)

    elif isinstance(universe, dict):
        # if dict: keys are tickers OR dict has 'tickers' field
        if "tickers" in universe and isinstance(universe["tickers"], (list, tuple, set)):
            tickers = [str(x).strip() for x in universe["tickers"] if str(x).strip()]
        else:
            tickers = [str(k).strip() for k in universe.keys() if str(k).strip()]
        for t in tickers:
            ticker_meta.setdefault(t, ("", ""))

    else:
        # unknown type: try to coerce single ticker
        meta = _coerce_meta_from_row(universe)
        tickers = []
        if meta:
            t, name, sector = meta
            tickers = [t]
            ticker_meta[t] = (name, sector)

    out = {"D": [], "W": [], "M": []}

    cfg = {
        "D": dict(min_len=60, max_len=220, min_depth=0.35, max_depth=0.65, rim_frac=0.15, min_progress=0.92),
        "W": dict(min_len=40, max_len=180, min_depth=0.45, max_depth=0.85, rim_frac=0.18, min_progress=0.92),
        "M": dict(min_len=18, max_len=80,  min_depth=0.55, max_depth=0.90, rim_frac=0.20, min_progress=0.95),
    }

    for tf in ("D", "W", "M"):
        met = []
        rule = "ME" if tf == "M" else tf  # pandas >=2.2 uses 'ME' for month-end
        for ticker in tickers:
            df = ohlc_map.get(ticker)
            if df is None or len(df) < 60:
                continue

            dfr = _resample(df, rule) if tf != "D" else df.copy()
            if dfr is None or len(dfr) < cfg[tf]["min_len"]:
                continue

            m = _saucer_score(
                dfr,
                min_len=cfg[tf]["min_len"],
                max_len=cfg[tf]["max_len"],
                rim_frac=cfg[tf]["rim_frac"],
                min_depth=cfg[tf]["min_depth"],
                max_depth=cfg[tf]["max_depth"],
                min_progress=cfg[tf]["min_progress"],
            )
            if not m:
                continue

            name, sector = ticker_meta.get(ticker, ("", ""))
            met.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "timeframe": "日足" if tf == "D" else ("週足" if tf == "W" else "月足"),
                "entry_rim": float(m["rim_price"]),
                "progress": float(m["progress"]),
                "depth": float(m["depth_pct"]),
            })

        met.sort(key=lambda x: (x["progress"], x["depth"]), reverse=True)
        out[tf] = met[:max_each]

    return out
