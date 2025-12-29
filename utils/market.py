from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf


def _hist(symbol: str, days: int = 80) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=f"{days}d", auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.rolling(n).mean() / (dn.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))


def _score_from_index(close: pd.Series) -> pd.Series:
    """0-100。Trend/Momentum/Risk を簡易合成。"""
    if close is None or len(close) < 60:
        return pd.Series(dtype=float)

    c = close.astype(float)
    sma20 = _sma(c, 20)
    sma50 = _sma(c, 50)
    sma10 = _sma(c, 10)

    r5 = c.pct_change(5)
    r20 = c.pct_change(20)
    rsi14 = _rsi(c, 14)

    # Risk: 短期ボラ上昇、急落
    ret = c.pct_change()
    vola20 = ret.rolling(20).std()
    drop = ret.rolling(2).sum()  # 2日落ち

    score = pd.Series(50.0, index=c.index)

    # Trend
    trend = (c > sma50).astype(float) + (sma20 > sma50).astype(float)
    score += (trend - 1.0) * 12.0

    # Momentum
    score += np.clip(r5 * 100 * 0.7, -10, 10)
    score += np.clip(r20 * 100 * 0.4, -8, 8)
    score += np.clip((rsi14 - 50) * 0.20, -6, 6)

    # Risk
    score -= np.clip((vola20 - vola20.rolling(60).mean()) * 200, -0, 10)
    score -= np.clip((-drop) * 250, 0, 12)

    # Close vs SMA10（短期の警戒）
    score -= ((c < sma10) & (c > 0)).astype(float) * 6.0

    return score.clip(0, 100)


def calc_market_score() -> Dict:
    """TOPIX/N225の合成で MarketScore と Δ3d を返す。"""
    topx = _hist("^TOPX", days=120)
    n225 = _hist("^N225", days=120)

    if topx.empty and n225.empty:
        return {"score": 50, "delta3d": 0, "comment": "中立"}

    scores: List[pd.Series] = []
    if not topx.empty:
        scores.append(_score_from_index(topx["Close"]))
    if not n225.empty:
        scores.append(_score_from_index(n225["Close"]))

    s = pd.concat(scores, axis=1).mean(axis=1).dropna()
    if s.empty:
        return {"score": 50, "delta3d": 0, "comment": "中立"}

    score_now = int(np.clip(round(float(s.iloc[-1])), 0, 100))
    score_3d = float(s.iloc[-4]) if len(s) >= 4 else float(s.iloc[0])
    delta3d = int(round(score_now - score_3d))

    comment = "強め" if score_now >= 70 else ("やや強め" if score_now >= 60 else ("中立" if score_now >= 50 else ("弱め" if score_now >= 40 else "弱い")))

    return {"score": score_now, "delta3d": delta3d, "comment": comment}


def enhance_market_score() -> Dict:
    """MarketScore に SOX/NVDA など Risk-on/off を薄く反映。"""
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX
    try:
        sox = _hist("^SOX", days=10)
        if not sox.empty and len(sox) >= 6:
            chg = float(sox["Close"].iloc[-1] / sox["Close"].iloc[-6] - 1.0) * 100.0
            score += float(np.clip(chg / 2.0, -5.0, 5.0))
            mkt["sox_5d"] = chg
    except Exception:
        pass

    # NVDA
    try:
        nv = _hist("NVDA", days=10)
        if not nv.empty and len(nv) >= 6:
            chg = float(nv["Close"].iloc[-1] / nv["Close"].iloc[-6] - 1.0) * 100.0
            score += float(np.clip(chg / 3.0, -4.0, 4.0))
            mkt["nvda_5d"] = chg
    except Exception:
        pass

    mkt["score"] = int(np.clip(round(score), 0, 100))

    # delta3d は calc_market_score のまま（強化分は補助情報扱い）
    return mkt