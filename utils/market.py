from __future__ import annotations

import numpy as np
import yfinance as yf

def _hist(symbol: str, period: str = "6d"):
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

def _chg_pct(df) -> float:
    try:
        if df is None or len(df) < 2:
            return 0.0
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return 0.0

def _chg_1d_pct(symbol: str) -> float:
    df = _hist(symbol, period="3d")
    try:
        if df is None or len(df) < 2:
            return 0.0
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)
    except Exception:
        return 0.0

def calc_market_score() -> dict:
    nk = _chg_pct(_hist("^N225", "6d"))
    tp = _chg_pct(_hist("^TOPX", "6d"))

    base = 50.0 + float(np.clip((nk + tp) / 2.0, -20, 20))
    score = int(np.clip(round(base), 0, 100))

    if score >= 70:
        comment = "強い（順張りOK）"
    elif score >= 60:
        comment = "やや強い"
    elif score >= 50:
        comment = "中立"
    elif score >= 40:
        comment = "弱い"
    else:
        comment = "危険"

    return {"score": score, "comment": comment, "n225_5d": nk, "topix_5d": tp}

def enhance_market_score() -> dict:
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    sox_df = _hist("^SOX", "6d")
    if sox_df is not None and len(sox_df) >= 2:
        sox_5d = _chg_pct(sox_df)
        score += float(np.clip(sox_5d / 2.0, -5.0, 5.0))
        mkt["sox_5d"] = sox_5d

    nv_df = _hist("NVDA", "6d")
    if nv_df is not None and len(nv_df) >= 2:
        nv_5d = _chg_pct(nv_df)
        score += float(np.clip(nv_5d / 3.0, -4.0, 4.0))
        mkt["nvda_5d"] = nv_5d

    fut_sym_used = ""
    fut_1d = 0.0
    for sym in ("NKD=F", "NIY=F", "NK=F", "N225=F"):
        v = _chg_1d_pct(sym)
        if np.isfinite(v) and abs(v) > 0:
            fut_1d = float(v)
            fut_sym_used = sym
            break

    mkt["futures_1d"] = float(fut_1d)
    mkt["futures_symbol"] = fut_sym_used
    mkt["futures_risk_on"] = bool(np.isfinite(fut_1d) and fut_1d >= 1.0)

    mkt["score"] = int(np.clip(round(score), 0, 100))
    return mkt
