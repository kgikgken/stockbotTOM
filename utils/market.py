from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
import yfinance as yf

def _hist(symbol: str, period: str = "260d") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty or len(df) < 80:
            return None
        return df
    except Exception:
        return None

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan

def _score_index(df: pd.DataFrame) -> Dict[str, float]:
    close = df["Close"].astype(float)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    c = float(close.iloc[-1])
    m20 = float(ma20.iloc[-1])
    m50 = float(ma50.iloc[-1])
    m20_prev = float(ma20.iloc[-6]) if len(ma20) >= 6 else m20
    slope20 = (m20 / (m20_prev + 1e-9) - 1.0)

    # momentum
    chg5 = float(close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0 if len(close) >= 6 else 0.0
    chg20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0 if len(close) >= 21 else 0.0

    # gap frequency (risk)
    atr = _atr(df, 14)
    op = df["Open"].astype(float)
    pc = close.shift(1)
    gap = (op / (pc + 1e-9) - 1.0).abs()
    if np.isfinite(atr) and atr > 0:
        atr_pct = atr / c
        gap_thr = atr_pct * 0.8
        gap_freq = float((gap.tail(20) > gap_thr).mean()) if len(gap) >= 20 else float((gap > gap_thr).mean())
    else:
        gap_freq = float((gap.tail(20) > 0.02).mean()) if len(gap) >= 20 else float((gap > 0.02).mean())

    # vola
    ret = close.pct_change(fill_method=None)
    vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 20 else 0.0

    # structure score
    s = 50.0
    if c > m20 > m50:
        s += 18
    elif c > m20:
        s += 10
    elif m20 > m50:
        s += 5

    s += np.clip(slope20 * 2000.0, -8, 8)
    s += np.clip(chg5 * 0.6, -10, 10)
    s += np.clip(chg20 * 0.25, -8, 8)

    # risk penalty
    s -= np.clip(gap_freq * 40.0, 0, 12)
    s -= np.clip(max(0.0, (vola20 - 0.018) * 800.0), 0, 10)

    return {"score": float(np.clip(s, 0, 100)), "gap_freq": gap_freq, "vola20": vola20, "chg5": chg5}

def rr_min_for_market(score: int) -> float:
    if score >= 70:
        return 1.8
    if score >= 60:
        return 2.0
    if score >= 50:
        return 2.2
    return 2.5

def leverage_for_market(score: int, macro_on: bool) -> float:
    # base
    if score >= 70:
        lev = 2.0
    elif score >= 60:
        lev = 1.7
    elif score >= 50:
        lev = 1.3
    elif score >= 40:
        lev = 1.1
    else:
        lev = 1.0
    # macro suppression
    if macro_on:
        lev = min(lev, 1.1)
    return float(lev)

def calc_market_context(today_date) -> Dict[str, object]:
    # Composite of N225 and TOPIX
    nk = _hist("^N225")
    tp = _hist("^TOPX")
    if nk is None or tp is None:
        score = 50
        delta3 = 0.0
        return {
            "score": score,
            "delta3": delta3,
            "comment": "中立",
            "regime": "中立",
            "rr_min": rr_min_for_market(score),
            "adjev_min": 0.50,
            "rday_min": 0.50,
            "lev": 1.0,
        }

    s1 = _score_index(nk)["score"]
    s2 = _score_index(tp)["score"]
    score = int(np.clip(round((s1 + s2) / 2.0), 0, 100))

    # delta3: recompute using truncated data (approx. 3 business days)
    cut = 4  # ~3 business days
    nk3 = nk.iloc[:-cut] if len(nk) > cut + 60 else nk
    tp3 = tp.iloc[:-cut] if len(tp) > cut + 60 else tp
    s1_3 = _score_index(nk3)["score"]
    s2_3 = _score_index(tp3)["score"]
    score_3 = int(np.clip(round((s1_3 + s2_3) / 2.0), 0, 100))
    delta3 = float(score - score_3)

    if score >= 70:
        regime = "強い（順張りOK）"
    elif score >= 60:
        regime = "やや強い（順張り中心）"
    elif score >= 50:
        regime = "中立（押し目中心）"
    elif score >= 45:
        regime = "弱め（新規絞る）"
    else:
        regime = "弱い（新規NG）"

    return {
        "score": score,
        "delta3": delta3,
        "comment": regime,
        "regime": regime,
        "rr_min": rr_min_for_market(score),
        "adjev_min": 0.50,
        "rday_min": 0.50,
        "lev": 1.0,  # filled in report using macro flag
    }
