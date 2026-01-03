# utils/market.py
from __future__ import annotations

from typing import Dict

import numpy as np
import yfinance as yf

# ============================================================
# Helpers
# ============================================================
def _fetch(symbol: str, period: str = "80d"):
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _pct(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return 0.0
    return float((a / b - 1.0) * 100.0)


def _sma(series, n: int):
    if series is None or len(series) < n:
        return np.nan
    return float(series.rolling(n).mean().iloc[-1])


# ============================================================
# Market Regime
# ============================================================
def calc_market_score() -> Dict:
    """
    0-100 の MarketScore（短期スイング用）
    構成：
      - Trend: Close > SMA20 > SMA50
      - Momentum: 5d / 20d リターン
      - Risk: 直近ボラ上昇・急落の抑制
    """
    idx = _fetch("^TOPX")  # TOPIX 優先
    if idx is None or len(idx) < 60:
        return {"score": 50, "comment": "中立", "detail": {}}

    close = idx["Close"].astype(float)
    c = float(close.iloc[-1])
    c5 = float(close.iloc[-6]) if len(close) >= 6 else c
    c20 = float(close.iloc[-21]) if len(close) >= 21 else c

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)

    score = 50.0
    detail = {}

    # --- Trend ---
    if np.isfinite(sma20) and np.isfinite(sma50):
        if c > sma20 > sma50:
            score += 10
            detail["trend"] = "up"
        elif c < sma20 < sma50:
            score -= 10
            detail["trend"] = "down"
        else:
            detail["trend"] = "side"

    # --- Momentum ---
    r5 = _pct(c, c5)
    r20 = _pct(c, c20)
    score += np.clip((r5 + r20) * 0.5, -10, 10)
    detail["r5"] = r5
    detail["r20"] = r20

    # --- Risk (vol expansion / drawdown) ---
    ret = close.pct_change(fill_method=None)
    vola20 = ret.rolling(20).std().iloc[-1]
    vola5 = ret.rolling(5).std().iloc[-1]
    if np.isfinite(vola20) and np.isfinite(vola5) and vola5 > vola20 * 1.6:
        score -= 5
        detail["risk"] = "vola_up"

    # clamp
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
        "detail": detail,
    }


def enhance_market_score(prev_score: int | None = None) -> Dict:
    """
    calc_market_score + 変化速度（Δ3d）
    """
    mkt = calc_market_score()
    score = mkt["score"]

    # Δ3d（3営業日前との差）
    idx = _fetch("^TOPX", period="20d")
    delta3 = 0
    if idx is not None and len(idx) >= 4:
        c = float(idx["Close"].iloc[-1])
        c3 = float(idx["Close"].iloc[-4])
        delta3 = int(np.clip(round(_pct(c, c3)), -20, 20))

    mkt["delta3d"] = delta3

    # NO-TRADE 判定用フラグ
    mkt["no_trade"] = bool(
        score < 45 or (delta3 <= -5 and score < 55)
    )

    return mkt