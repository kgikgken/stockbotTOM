from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import clamp


def _rsi14(close: pd.Series) -> float:
    if close is None or len(close) < 20:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]
    rs = float(avg_gain) / (float(avg_loss) + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if not np.isfinite(rsi):
        return 50.0
    return float(rsi)


def _fetch_index(symbol: str, period: str = "120d") -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _score_from_close(close: pd.Series) -> int:
    # 0-100
    if close is None or len(close) < 60:
        return 50

    c = close.astype(float)
    last = float(c.iloc[-1])
    sma20 = float(c.rolling(20).mean().iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])

    # momentum
    r5 = float(c.iloc[-1] / c.iloc[-6] - 1.0) * 100.0 if len(c) >= 6 else 0.0
    r20 = float(c.iloc[-1] / c.iloc[-21] - 1.0) * 100.0 if len(c) >= 21 else 0.0
    rsi = _rsi14(c)

    base = 50.0

    # trend component
    if last > sma50:
        base += 10.0
    else:
        base -= 10.0

    if sma20 > sma50:
        base += 10.0
    else:
        base -= 10.0

    # momentum component
    base += clamp(r5 / 2.0, -8.0, 8.0)
    base += clamp(r20 / 4.0, -8.0, 8.0)

    # rsi stability
    if 45 <= rsi <= 65:
        base += 6.0
    elif rsi < 35:
        base -= 8.0
    elif rsi > 75:
        base -= 6.0

    score = int(round(clamp(base, 0.0, 100.0)))
    return score


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


def _recommend_leverage(score: int) -> Tuple[float, str]:
    if score >= 70:
        return 2.0, "強気（押し目＋一部ブレイク）"
    if score >= 60:
        return 1.7, "やや強気（押し目メイン）"
    if score >= 50:
        return 1.3, "中立（厳選・押し目中心）"
    if score >= 40:
        return 1.1, "やや守り（新規ロット小さめ）"
    return 1.0, "守り（新規かなり絞る）"


def build_market_context() -> Dict:
    # Use TOPIX primarily; fallback N225
    topx = _fetch_index("^TOPX", period="160d")
    n225 = _fetch_index("^N225", period="160d")

    # Choose series
    close = None
    if topx is not None and not topx.empty and "Close" in topx.columns:
        close = topx["Close"]
        idx_name = "TOPIX"
    elif n225 is not None and not n225.empty and "Close" in n225.columns:
        close = n225["Close"]
        idx_name = "N225"
    else:
        score = 50
        delta3d = 0
        lev, lev_comment = _recommend_leverage(score)
        return {
            "index": "N/A",
            "score": score,
            "comment": _comment(score),
            "delta3d": delta3d,
            "lev": lev,
            "lev_comment": lev_comment,
            "allow_new": True,
            "no_trade_reasons": [],
            "regime_multiplier": 1.0,
        }

    score_today = _score_from_close(close)

    # Delta3d: compute score 3 trading days ago (slice)
    if len(close) >= 8:
        close_past = close.iloc[:-3]
        score_past = _score_from_close(close_past)
        delta3d = int(score_today - score_past)
    else:
        delta3d = 0

    # NO-TRADE mechanical
    reasons = []
    allow_new = True
    if score_today < 45:
        allow_new = False
        reasons.append("MarketScore<45")
    if delta3d <= -5 and score_today < 55:
        allow_new = False
        reasons.append("Δ3d<=-5 & MarketScore<55")

    # Regime multiplier for AdjEV
    mult = 1.0
    if score_today >= 60 and delta3d >= 0:
        mult = 1.05
    if delta3d <= -5:
        mult *= 0.70
    if score_today < 45:
        mult *= 0.65
    mult = float(clamp(mult, 0.50, 1.10))

    lev, lev_comment = _recommend_leverage(score_today)

    return {
        "index": idx_name,
        "score": int(score_today),
        "comment": _comment(int(score_today)),
        "delta3d": int(delta3d),
        "lev": float(lev),
        "lev_comment": lev_comment,
        "allow_new": bool(allow_new),
        "no_trade_reasons": reasons,
        "regime_multiplier": mult,
    }