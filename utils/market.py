from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

INDEX_N225 = "^N225"
INDEX_TOPIX = "^TOPX"


def _fetch(symbol: str, period: str = "200d") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    return df if df is not None else pd.DataFrame()


def _rsi14(close: pd.Series) -> float:
    if close is None or len(close) < 20:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def _score_from_df(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) < 80:
        return {"score": 50, "comment": "中立"}

    c = df["Close"].astype(float)
    ret = c.pct_change(fill_method=None)

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()

    close = float(c.iloc[-1])
    s20 = float(sma20.iloc[-1])
    s50 = float(sma50.iloc[-1])

    rsi = _rsi14(c)
    vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 25 else float("nan")
    dd3 = float((c.iloc[-1] / c.iloc[-4] - 1.0) * 100.0) if len(c) >= 4 else 0.0

    sc = 50.0

    if np.isfinite(s20) and np.isfinite(s50) and np.isfinite(close):
        sc += 8 if close > s50 else -8
        sc += 10 if s20 > s50 else -10

    if np.isfinite(rsi):
        if rsi >= 60:
            sc += 10
        elif rsi >= 50:
            sc += 5
        elif rsi <= 40:
            sc -= 10
        elif rsi <= 45:
            sc -= 5

    if np.isfinite(dd3):
        if dd3 <= -2.5:
            sc -= 12
        elif dd3 <= -1.5:
            sc -= 7

    if np.isfinite(vola20):
        if vola20 >= 0.025:
            sc -= 8
        elif vola20 <= 0.015:
            sc += 3

    score = int(np.clip(round(sc), 0, 100))
    comment = (
        "強め" if score >= 70 else
        "やや強め" if score >= 60 else
        "中立" if score >= 50 else
        "弱め" if score >= 40 else
        "弱い"
    )
    return {"score": score, "comment": comment}


def calc_market_score() -> dict:
    df = _fetch(INDEX_TOPIX)
    if df is None or df.empty:
        df = _fetch(INDEX_N225)
    return _score_from_df(df)


def enhance_market_score() -> dict:
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    def _five_day_chg(symbol: str) -> float:
        try:
            d = yf.Ticker(symbol).history(period="6d", auto_adjust=True)
            if d is None or d.empty or len(d) < 2:
                return 0.0
            c = d["Close"].astype(float)
            return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
        except Exception:
            return 0.0

    sox = _five_day_chg("^SOX")
    nvda = _five_day_chg("NVDA")

    score += float(np.clip(sox / 2.0, -5.0, 5.0))
    score += float(np.clip(nvda / 3.0, -4.0, 4.0))

    mkt["score"] = int(np.clip(round(score), 0, 100))
    return mkt


def calc_market_score_3d_delta() -> int:
    df = _fetch(INDEX_TOPIX)
    if df is None or df.empty:
        df = _fetch(INDEX_N225)

    if df is None or df.empty or len(df) < 90:
        return 0

    df_today = df.copy()
    df_past = df.iloc[:-3].copy()

    s_today = _score_from_df(df_today).get("score", 50)
    s_past = _score_from_df(df_past).get("score", 50)
    return int(s_today - s_past)
