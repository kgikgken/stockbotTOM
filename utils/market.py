from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def _hist(symbol: str, period: str = "200d") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _sma(series: pd.Series, window: int) -> float:
    if series is None or len(series) < window:
        return float(series.iloc[-1])
    return float(series.rolling(window).mean().iloc[-1])


def _slope(series: pd.Series, window: int = 5) -> float:
    if series is None or len(series) < window + 1:
        return 0.0
    x = series.iloc[-(window + 1) :].astype(float)
    # simple slope of pct change
    base = float(x.iloc[0])
    if not np.isfinite(base) or base == 0:
        return 0.0
    return float((float(x.iloc[-1]) / base - 1.0) / window)


def _gap_freq(df: pd.DataFrame, window: int = 30) -> float:
    if df is None or len(df) < window + 2:
        return 0.0
    sub = df.tail(window)
    o = sub["Open"].astype(float)
    c = sub["Close"].astype(float)
    prev_c = c.shift(1)
    gap = (o - prev_c).abs() / (prev_c.replace(0, np.nan)).abs()
    return float(np.nanmean((gap > 0.01).astype(float)))  # >1% gaps


def calc_market_score() -> Dict:
    """MarketScore 0-100.

    Inputs:
      - Index MA structure (20/50)
      - short momentum (5d)
      - volatility (20d)
      - gap frequency
    """
    nk = _hist("^N225")
    tp = _hist("^TOPX")
    if nk is None or tp is None:
        return {"score": 50, "comment": "中立"}

    def score_one(df: pd.DataFrame) -> float:
        close = df["Close"].astype(float)
        c = float(close.iloc[-1])
        sma20 = _sma(close, 20)
        sma50 = _sma(close, 50)

        sc = 50.0

        # MA structure
        if c > sma20 > sma50:
            sc += 12
        elif c > sma20:
            sc += 6
        elif sma20 > sma50:
            sc += 3
        else:
            sc -= 6

        # slope of SMA20
        if len(close) >= 25:
            sma20_series = close.rolling(20).mean()
            sl = _slope(sma20_series.dropna(), window=5)
            sc += float(np.clip(sl * 3000, -8, 8))

        # momentum 5d
        if len(close) >= 6:
            m5 = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0)
            sc += float(np.clip(m5, -8, 8))

        # volatility
        ret = close.pct_change(fill_method=None)
        vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 25 else 0.01
        # too high vola -> penalty
        sc += float(np.clip((0.02 - vola20) * 400, -8, 6))

        # gap frequency
        gf = _gap_freq(df, window=30)
        sc -= float(np.clip(gf * 15, 0, 6))

        return float(np.clip(sc, 0, 100))

    nk_sc = score_one(nk)
    tp_sc = score_one(tp)

    score = float(np.mean([nk_sc, tp_sc]))
    score_i = int(np.clip(round(score), 0, 100))

    if score_i >= 70:
        comment = "強い"
    elif score_i >= 60:
        comment = "やや強い"
    elif score_i >= 50:
        comment = "中立"
    elif score_i >= 40:
        comment = "弱い"
    else:
        comment = "弱すぎ"

    return {
        "score": score_i,
        "comment": comment,
        "n225_score": float(nk_sc),
        "topix_score": float(tp_sc),
    }


def futures_snapshot() -> Dict:
    """Return Nikkei futures change (NKD=F) 1d pct if available."""
    out: Dict = {}
    df = _hist("NKD=F", period="6d")
    if df is None or len(df) < 2:
        return out
    c = df["Close"].astype(float)
    chg = float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    out["futures_symbol"] = "NKD=F"
    out["futures_5d"] = chg
    # also most recent 1d
    if len(c) >= 2:
        out["futures_1d"] = float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)
    return out


def market_summary(prev_scores: Tuple[int, int, int] | None = None) -> Dict:
    """Compute market summary and 3d delta.

    prev_scores: last 3 days scores (t-1, t-2, t-3). If provided, delta_3d uses t-3.
    """
    mkt = calc_market_score()
    score = int(mkt.get("score", 50))

    delta_3d = 0.0
    if prev_scores and len(prev_scores) >= 3:
        delta_3d = float(score - int(prev_scores[2]))

    fut = futures_snapshot()
    mkt.update(fut)
    mkt["delta_3d"] = delta_3d

    # Risk-ON heuristic (futures strong up)
    risk_on = False
    if "futures_1d" in fut and np.isfinite(float(fut["futures_1d"])):
        risk_on = float(fut["futures_1d"]) >= 1.0
    mkt["risk_on"] = bool(risk_on)
    return mkt
