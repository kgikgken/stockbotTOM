from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd

from .util import rsi14, atr14, safe_float
from utils.features import fetch_history


@dataclass
class MarketInfo:
    score: int
    fut_pct: float
    dscore_3d: float
    risk_on: bool
    index_symbol: str
    vol_atr14: float


def _fetch_index_candidates() -> List[str]:
    # Prefer TOPIX proxy tickers that are more reliable in yfinance.
    return ["^TOPX", "1306.T", "1321.T", "^N225", "NKD=F"]


def _get_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
    if col is None:
        return pd.Series(dtype=float)
    return df[col].astype(float)


def compute_market_info() -> MarketInfo:
    """Compute a lightweight market regime snapshot.

    NOTE: In the latest spec, MarketScore is used for **exit speed control only**.
    This function is intentionally simple and robust to missing data.
    """
    sym_used = ""
    df_used = None
    for sym in _fetch_index_candidates():
        try:
            df = fetch_history(sym, days=260)
            if df is not None and not df.empty:
                c = _get_close(df)
                if len(c) >= 30 and c.notna().sum() >= 25:
                    sym_used = sym
                    df_used = df
                    break
        except Exception:
            continue

    if df_used is None:
        return MarketInfo(score=50, fut_pct=0.0, dscore_3d=0.0, risk_on=False, index_symbol="", vol_atr14=0.0)

    close = _get_close(df_used)
    rsi = safe_float(rsi14(close).iloc[-1], 50.0)
    vol = safe_float(atr14(df_used).iloc[-1], 0.0)

    # Normalize RSI to a 0-100 score (RSI itself is 0-100).
    score = int(max(0, min(100, round(rsi))))

    # Simple 3-day delta proxy
    dscore_3d = safe_float(rsi - safe_float(rsi14(close).iloc[-4], rsi), 0.0)

    # Futures: use NKD=F if available
    fut_pct = 0.0
    try:
        fdf = fetch_history("NKD=F", days=10)
        if fdf is not None and not fdf.empty:
            fc = _get_close(fdf)
            if len(fc) >= 2:
                fut_pct = safe_float((fc.iloc[-1] / fc.iloc[-2] - 1.0) * 100.0, 0.0)
    except Exception:
        fut_pct = 0.0

    risk_on = fut_pct > 1.0

    return MarketInfo(score=score, fut_pct=fut_pct, dscore_3d=dscore_3d, risk_on=risk_on, index_symbol=sym_used, vol_atr14=vol)


def compute_market_score() -> int:
    return compute_market_info().score
