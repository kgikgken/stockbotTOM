from __future__ import annotations

import numpy as np
import yfinance as yf

def _hist_close(symbol: str, days: int = 60):
    try:
        df = yf.Ticker(symbol).history(period=f"{days}d", auto_adjust=True)
        if df is None or df.empty:
            return None
        return df["Close"].astype(float)
    except Exception:
        return None

def _rsi14(close):
    if close is None or len(close) < 15:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def _score_index(close):
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(20).mean()
    c = float(close.iloc[-1])
    m20 = float(ma20.iloc[-1])
    m50 = float(ma50.iloc[-1])

    trend = 0.0
    if c > m50 and m20 > m50:
        trend = 1.0
    elif c > m50:
        trend = 0.7
    elif m20 > m50:
        trend = 0.5

    r5 = float(close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0 if len(close) >= 6 else 0.0
    r20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0 if len(close) >= 21 else 0.0
    rsi = _rsi14(close)

    mom = 0.5
    mom += np.clip((r5 + 0.7 * r20) / 20.0, -0.5, 0.5)
    if np.isfinite(rsi):
        mom += np.clip((rsi - 50) / 40.0, -0.4, 0.4)

    ret = close.pct_change(fill_method=None)
    vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 21 else 0.02
    risk = 0.5 - np.clip((vola20 - 0.015) / 0.05, 0.0, 0.5)

    raw = 50.0 + 20.0 * (trend - 0.5) + 25.0 * (mom - 0.5) + 15.0 * (risk - 0.25)
    return float(np.clip(raw, 0.0, 100.0))

def enhance_market_score() -> dict:
    tp = _hist_close("^TOPX", 80)
    nk = _hist_close("^N225", 80)

    if tp is None and nk is None:
        return {"score": 50, "comment": "中立", "delta3d": 0}

    scores = []
    if tp is not None and len(tp) >= 25:
        scores.append(_score_index(tp))
    if nk is not None and len(nk) >= 25:
        scores.append(_score_index(nk))

    base = float(np.mean(scores)) if scores else 50.0

    def score_shift(close, shift: int):
        if close is None or len(close) < 25 + shift:
            return np.nan
        sub = close.iloc[:-shift]
        if len(sub) < 25:
            return np.nan
        return _score_index(sub)

    deltas = []
    if tp is not None and len(tp) >= 28:
        deltas.append(_score_index(tp) - score_shift(tp, 3))
    if nk is not None and len(nk) >= 28:
        deltas.append(_score_index(nk) - score_shift(nk, 3))
    deltas = [d for d in deltas if np.isfinite(d)]
    delta3d = int(round(float(np.mean(deltas)))) if deltas else 0

    score = float(base)

    def five_day_chg(symbol: str) -> float:
        c = _hist_close(symbol, 12)
        if c is None or len(c) < 6:
            return 0.0
        return float((c.iloc[-1] / c.iloc[-6] - 1.0) * 100.0)

    try:
        score += float(np.clip(five_day_chg("^SOX") / 2.0, -5.0, 5.0))
    except Exception:
        pass
    try:
        score += float(np.clip(five_day_chg("NVDA") / 3.0, -4.0, 4.0))
    except Exception:
        pass

    score_i = int(np.clip(round(score), 0, 100))
    comment = "強め" if score_i >= 70 else "やや強め" if score_i >= 60 else "中立" if score_i >= 50 else "弱め" if score_i >= 40 else "弱い"
    return {"score": score_i, "comment": comment, "delta3d": delta3d}
