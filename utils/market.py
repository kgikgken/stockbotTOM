from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import clamp


def _fetch_index(symbol: str, period: str = "80d") -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _rsi(close: pd.Series, period: int = 14) -> float:
    if close is None or len(close) < period + 2:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    v = float(rsi.iloc[-1])
    return v if np.isfinite(v) else float("nan")


def _score_from_index(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    return: (score_component, ret5, ret20, rsi14)
    """
    if df is None or df.empty or len(df) < 55:
        return 0.0, 0.0, 0.0, 50.0

    close = df["Close"].astype(float)
    c = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])

    ret5 = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0) if len(close) >= 6 else 0.0
    ret20 = float((close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0) if len(close) >= 21 else 0.0
    rsi14 = _rsi(close, 14)

    sc = 0.0
    # Trend
    if c > ma50:
        sc += 10
    if ma20 > ma50:
        sc += 10
    # Momentum
    sc += clamp(ret5, -6, 6) * 1.5
    sc += clamp(ret20, -12, 12) * 0.8
    # RSI (過熱/弱さ)
    if np.isfinite(rsi14):
        if 45 <= rsi14 <= 65:
            sc += 6
        elif rsi14 < 40:
            sc -= 6
        elif rsi14 > 75:
            sc -= 4

    return sc, ret5, ret20, (float(rsi14) if np.isfinite(rsi14) else 50.0)


def build_market_context() -> Dict:
    """
    MarketScore(0-100) と Δ3d を作る
    """
    n225 = _fetch_index("^N225")
    topx = _fetch_index("^TOPX")

    sc1, n225_5d, n225_20d, n225_rsi = _score_from_index(n225)
    sc2, topx_5d, topx_20d, topx_rsi = _score_from_index(topx)

    base = 50.0 + (sc1 + sc2) / 2.0
    score = int(clamp(round(base), 0, 100))

    # Δ3d（スコア変化）
    # “3日前”のスコアを再計算して差を取る（粗いが安定）
    delta3d = 0
    try:
        if n225 is not None and len(n225) >= 10 and topx is not None and len(topx) >= 10:
            n225_past = n225.iloc[:-3]
            topx_past = topx.iloc[:-3]
            sc1p, _, _, _ = _score_from_index(n225_past)
            sc2p, _, _, _ = _score_from_index(topx_past)
            past = int(clamp(round(50.0 + (sc1p + sc2p) / 2.0), 0, 100))
            delta3d = int(score - past)
    except Exception:
        delta3d = 0

    # コメント
    if score >= 70:
        comment = "強い"
        phase = "上昇基調"
    elif score >= 60:
        comment = "やや強い"
        phase = "上昇基調"
    elif score >= 50:
        comment = "中立"
        phase = "不安定"
    elif score >= 45:
        comment = "弱い"
        phase = "下落警戒"
    else:
        comment = "かなり弱い"
        phase = "下落警戒"

    return {
        "score": score,
        "delta3d": int(delta3d),
        "comment": comment,
        "phase": phase,
        "n225_5d": float(n225_5d),
        "topix_5d": float(topx_5d),
        "n225_20d": float(n225_20d),
        "topix_20d": float(topx_20d),
        "n225_rsi": float(n225_rsi),
        "topix_rsi": float(topx_rsi),
    }