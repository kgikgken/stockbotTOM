from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


def _fetch_close(symbol: str, period: str = "40d") -> pd.Series:
    df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df["Close"].astype(float)


def _chg_5d_at(close: pd.Series, end_iloc: int) -> float:
    if close is None or len(close) < 7:
        return 0.0
    i = end_iloc if end_iloc >= 0 else len(close) + end_iloc
    if i - 5 < 0 or i >= len(close):
        return 0.0
    a = float(close.iloc[i - 5])
    b = float(close.iloc[i])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return 0.0
    return float((b / a - 1.0) * 100.0)


def _score_from_5d(nk_5d: float, tp_5d: float) -> int:
    base = 50.0 + float(np.clip((nk_5d + tp_5d) / 2.0, -20, 20))
    return int(np.clip(round(base), 0, 100))


def _comment(score: int) -> str:
    if score >= 70:
        return "強め"
    if score >= 60:
        return "やや強め"
    if score >= 50:
        return "中立"
    if score >= 40:
        return "弱め"
    return "弱い"


def calc_market_score() -> Dict[str, object]:
    nk = _fetch_close("^N225")
    tp = _fetch_close("^TOPX")
    nk_5d = _chg_5d_at(nk, -1)
    tp_5d = _chg_5d_at(tp, -1)
    score = _score_from_5d(nk_5d, tp_5d)
    return {"score": int(score), "comment": _comment(int(score)), "n225_5d": float(nk_5d), "topix_5d": float(tp_5d)}


def calc_delta_market_score_3d() -> int:
    nk = _fetch_close("^N225")
    tp = _fetch_close("^TOPX")
    s_today = _score_from_5d(_chg_5d_at(nk, -1), _chg_5d_at(tp, -1))
    s_3d = _score_from_5d(_chg_5d_at(nk, -4), _chg_5d_at(tp, -4))
    return int(s_today - s_3d)


def enhance_market_score() -> Dict[str, object]:
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    def _five_day_chg(symbol: str) -> float:
        close = _fetch_close(symbol, period="15d")
        return _chg_5d_at(close, -1)

    try:
        sox = _five_day_chg("^SOX")
        score += float(np.clip(sox / 2.0, -5.0, 5.0))
        mkt["sox_5d"] = float(sox)
    except Exception:
        pass

    try:
        nv = _five_day_chg("NVDA")
        score += float(np.clip(nv / 3.0, -4.0, 4.0))
        mkt["nvda_5d"] = float(nv)
    except Exception:
        pass

    mkt["score"] = int(np.clip(round(score), 0, 100))
    mkt["comment"] = _comment(int(mkt["score"]))

    try:
        mkt["d_market_3d"] = int(calc_delta_market_score_3d())
    except Exception:
        mkt["d_market_3d"] = 0

    return mkt
