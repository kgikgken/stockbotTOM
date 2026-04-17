from __future__ import annotations

from typing import Dict, Tuple, Iterable, Optional

import os
import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import sma, returns, safe_float, clamp, download_history_bulk

# NOTE:
#  - yfinance does *not* reliably provide "^TOPX" (TOPIX index).
#  - For Japan, TOPIX ETF tickers (e.g. 1306.T) are far more stable.
#  - We keep a fallback list to avoid noisy "possibly delisted" warnings.


def _parse_csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    xs = [x.strip() for x in str(raw).split(",") if x.strip()]
    return xs


def _fetch_index(symbol: str, period: str = "260d") -> pd.DataFrame:
    """Fetch an index/ETF series resiliently.

    Uses utils.util.download_history_bulk() to inherit retry/backoff and caching.
    """
    try:
        m = download_history_bulk(
            [symbol],
            period=period,
            interval="1d",
            group_size=1,
            pause_sec=0.0,
            auto_adjust=True,
            min_bars=20,
        )
        df = m.get(symbol)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
def _fetch_any(symbols: Iterable[str], period: str = "260d") -> tuple[str, pd.DataFrame]:
    for sym in symbols:
        df = _fetch_index(sym, period=period)
        if df is not None and not df.empty and len(df) >= 60:
            return sym, df
    return "", pd.DataFrame()


def _ma_structure_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 60:
        return 0.0
    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    c_last = safe_float(c.iloc[-1], np.nan)
    m20 = safe_float(ma20.iloc[-1], np.nan)
    m50 = safe_float(ma50.iloc[-1], np.nan)
    if not (np.isfinite(c_last) and np.isfinite(m20) and np.isfinite(m50)):
        return 0.0

    sc = 0.0
    if c_last > m20 > m50:
        sc += 12
    elif c_last > m20:
        sc += 6
    elif m20 > m50:
        sc += 3

    # slope of MA20 over 5 days (less noisy than 1-day pct_change)
    if len(ma20) >= 26 and np.isfinite(safe_float(ma20.iloc[-6], np.nan)):
        slope20 = safe_float(ma20.iloc[-1] / ma20.iloc[-6] - 1.0, 0.0)
    else:
        slope20 = safe_float(ma20.pct_change(fill_method=None).iloc[-1], 0.0)

    if slope20 >= 0.004:
        sc += 8
    elif slope20 > 0:
        sc += 4 + slope20 / 0.004 * 4
    else:
        sc += max(0.0, 4 + slope20 * 200)

    return float(clamp(sc, 0, 20))


def _momentum_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 21:
        return 0.0
    c = df["Close"].astype(float)
    r5 = safe_float(c.iloc[-1] / c.iloc[-6] - 1.0, 0.0) * 100.0
    r20 = safe_float(c.iloc[-1] / c.iloc[-21] - 1.0, 0.0) * 100.0
    sc = 0.0
    sc += clamp(r5, -6, 6) * 2.0
    sc += clamp(r20, -12, 12) * 1.0
    return float(clamp(sc, -25, 25))


def _vol_gap_penalty(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 30:
        return 0.0
    c = df["Close"].astype(float)
    o = df["Open"].astype(float)
    prev = c.shift(1)
    gap = ((o - prev).abs() / (prev + 1e-9))
    gap_freq = float((gap.tail(20) > 0.012).mean())
    vola = safe_float(returns(df).tail(20).std(), 0.0)
    pen = 0.0
    pen += gap_freq * 10.0
    pen += clamp((vola - 0.012) * 500.0, 0.0, 10.0)
    return float(-pen)


def market_score() -> Dict[str, float]:
    # Prefer ETFs for stability. Users can override via env vars.
    topx_syms = _parse_csv_env("MARKET_TOPX_TICKERS", "998405.T,1306.T,^TOPX")
    n225_syms = _parse_csv_env("MARKET_N225_TICKERS", "^N225,1321.T")

    _, n225 = _fetch_any(n225_syms)
    _, topx = _fetch_any(topx_syms)

    base = 50.0

    # If one index is missing, still compute from the other.
    if not n225.empty:
        base += _ma_structure_score(n225)
        base += _momentum_score(n225) * 0.6
        base += _vol_gap_penalty(n225) * 0.7
    if not topx.empty:
        base += _ma_structure_score(topx)
        base += _momentum_score(topx) * 0.6
        base += _vol_gap_penalty(topx) * 0.7

    score = int(clamp(round(base), 0, 100))

    if score >= 70:
        comment = "強い"
    elif score >= 60:
        comment = "やや強い"
    elif score >= 50:
        comment = "中立"
    elif score >= 40:
        comment = "弱い"
    else:
        comment = "かなり弱い"

    return {"score": float(score), "comment": comment}


def futures_risk_on() -> Tuple[bool, float]:
    """Simple risk-on proxy using Nikkei futures.

    Returns (risk_on, change_percent).
    """
    try:
        m = download_history_bulk(
            ["NKD=F"],
            period="6d",
            interval="1d",
            group_size=1,
            pause_sec=0.0,
            auto_adjust=True,
            min_bars=2,
        )
        df = m.get("NKD=F")
        if df is None or df.empty or len(df) < 2:
            return False, 0.0
        chg = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0
        return bool(chg >= 1.0), float(chg)
    except Exception:
        return False, 0.0
