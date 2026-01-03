from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import yfinance as yf

from utils.util import safe_float


def _history(symbol: str, period: str = "120d"):
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        return df
    except Exception:
        return None


def _ma(series, n: int) -> float:
    try:
        if series is None or len(series) < n:
            return safe_float(series.iloc[-1])
        return safe_float(series.rolling(n).mean().iloc[-1])
    except Exception:
        return float("nan")


def _rsi(close, n: int = 14) -> float:
    try:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return safe_float(rsi.iloc[-1])
    except Exception:
        return float("nan")


def _ret(close, n: int) -> float:
    try:
        if close is None or len(close) < n + 1:
            return 0.0
        return float((close.iloc[-1] / close.iloc[-1 - n] - 1.0) * 100.0)
    except Exception:
        return 0.0


def _five_day_change(symbol: str) -> float:
    df = _history(symbol, period="10d")
    if df is None or df.empty or len(df) < 6:
        return 0.0
    c = df["Close"].astype(float)
    return float((c.iloc[-1] / c.iloc[-6] - 1.0) * 100.0)


def _calc_market_score() -> Dict:
    """
    World-Strong-ish market regime:
      - Trend: Close vs SMA20/50
      - Momentum: 5d/20d return
      - RSI
    """
    # Japan indices
    n225 = "^N225"
    topx = "^TOPX"

    df1 = _history(n225, period="140d")
    df2 = _history(topx, period="140d")

    score = 50.0
    comment = "中立"
    delta3d = 0

    # fallback if insufficient
    if df1 is None or df1.empty or df2 is None or df2.empty:
        return {"score": 50, "comment": "中立", "delta3d": 0, "n225_5d": 0.0, "topix_5d": 0.0}

    c1 = df1["Close"].astype(float)
    c2 = df2["Close"].astype(float)

    # trend
    for c in (c1, c2):
        ma20 = _ma(c, 20)
        ma50 = _ma(c, 50)
        last = safe_float(c.iloc[-1])
        if np.isfinite(last) and np.isfinite(ma20) and np.isfinite(ma50):
            if last > ma20 > ma50:
                score += 6
            elif last > ma50:
                score += 3
            else:
                score -= 5

    # momentum
    mom5 = (_ret(c1, 5) + _ret(c2, 5)) / 2.0
    mom20 = (_ret(c1, 20) + _ret(c2, 20)) / 2.0
    score += float(np.clip(mom5, -6, 6))
    score += float(np.clip(mom20 / 2.0, -6, 6))

    # RSI
    rsi1 = _rsi(c1, 14)
    rsi2 = _rsi(c2, 14)
    rsi = np.nanmean([rsi1, rsi2])
    if np.isfinite(rsi):
        if rsi >= 60:
            score += 4
        elif rsi <= 40:
            score -= 4

    # ΔMarketScore_3d（粗く：5d変化から3dを近似）
    # ※厳密にやるなら過去スコア履歴が必要だが、ここは軽量で。
    delta3d = int(np.clip(round(mom5 * 0.6), -25, 25))

    score = int(np.clip(round(score), 0, 100))
    if score >= 70:
        comment = "強め"
    elif score >= 60:
        comment = "やや強め"
    elif score >= 50:
        comment = "中立"
    elif score >= 40:
        comment = "弱め"
    else:
        comment = "弱い"

    return {
        "score": score,
        "comment": comment,
        "delta3d": int(delta3d),
        "n225_5d": float(_five_day_change(n225)),
        "topix_5d": float(_five_day_change(topx)),
    }


def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.7, "やや強気（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（厳選・押し目中心）"
    if mkt_score >= 40:
        return 1.1, "やや守り（新規ロット小さめ）"
    return 1.0, "守り（新規かなり絞る）"


def build_market_context(today_date) -> Dict:
    m = _calc_market_score()
    lev, lev_comment = recommend_leverage(int(m["score"]))
    m["lev"] = float(lev)
    m["lev_comment"] = str(lev_comment)
    return m