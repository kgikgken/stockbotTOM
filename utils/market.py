from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import clamp
from utils.events import build_event_warnings


def _history(symbol: str, period: str = "260d") -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        for c in ("Open", "High", "Low", "Close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        return df
    except Exception:
        return pd.DataFrame()


def _rsi(close: pd.Series, period: int = 14) -> float:
    if close is None or len(close) < period + 5:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    v = float(rsi.iloc[-1])
    return v


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 3:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = float(tr.rolling(period).mean().iloc[-1])
    return v


def calc_market_score() -> Dict:
    """
    0-100
    Trend + Momentum + Risk を最低限反映
    """
    topix = _history("^TOPX")
    nikkei = _history("^N225")

    if topix.empty and nikkei.empty:
        return {"score": 50, "comment": "中立", "delta3d": 0}

    # 指数はTOPIX優先、なければN225
    df = topix if not topix.empty else nikkei
    close = df["Close"].astype(float)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma10 = close.rolling(10).mean()

    c = float(close.iloc[-1])
    s20 = float(sma20.iloc[-1]) if len(sma20) else c
    s50 = float(sma50.iloc[-1]) if len(sma50) else c
    s10 = float(sma10.iloc[-1]) if len(sma10) else c

    rsi14 = _rsi(close, 14)
    atr14 = _atr(df, 14)
    atrp = float(atr14 / c) if np.isfinite(atr14) and c > 0 else 0.02

    # Momentum
    ret5 = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0) if len(close) >= 6 else 0.0
    ret20 = float((close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0) if len(close) >= 21 else 0.0

    # delta3d: score差分（後で score推定してから引くのは循環なので、指数リターン3日で代替）
    ret3 = float((close.iloc[-1] / close.iloc[-4] - 1.0) * 100.0) if len(close) >= 4 else 0.0

    score = 50.0

    # Trend
    if np.isfinite(c) and np.isfinite(s20) and np.isfinite(s50):
        if c > s50 and s20 > s50:
            score += 18
        elif c > s50:
            score += 10
        elif c < s50:
            score -= 10

    # Momentum
    score += clamp(ret5, -6, 6) * 1.2
    score += clamp(ret20, -10, 10) * 0.6

    # RSI（過熱/弱すぎ）
    if np.isfinite(rsi14):
        if rsi14 >= 70:
            score -= 6
        elif rsi14 <= 35:
            score -= 6
        else:
            score += 2

    # Risk（ボラ急増・s10割れ）
    if np.isfinite(atrp):
        if atrp >= 0.03:
            score -= 4
        if atrp >= 0.05:
            score -= 6

    if np.isfinite(s10) and np.isfinite(c) and c < s10:
        score -= 4

    score = int(clamp(round(score), 0, 100))

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

    # “ΔMarketScore_3d” は指数3日リターンから換算（見た目の整合のため）
    delta3d = int(clamp(round(ret3 * 3.0), -30, 30))

    return {
        "score": score,
        "comment": comment,
        "delta3d": delta3d,
        "index_ret5": ret5,
        "index_ret20": ret20,
        "index_rsi14": float(rsi14) if np.isfinite(rsi14) else None,
        "index_atr_pct": float(atrp) if np.isfinite(atrp) else None,
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


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


def no_trade_rule(mkt_score: int, delta3d: int) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    nt = False

    if mkt_score < 45:
        nt = True
        reasons.append("MarketScore<45")

    if (delta3d <= -5) and (mkt_score < 55):
        nt = True
        reasons.append("Δ3d<=-5 & MarketScore<55")

    return nt, reasons


def regime_multiplier(mkt_score: int, delta3d: int, is_event_risk_day: bool) -> float:
    mult = 1.00
    if mkt_score >= 60 and delta3d >= 0:
        mult *= 1.05
    if delta3d <= -5:
        mult *= 0.70
    if is_event_risk_day:
        mult *= 0.75
    return float(mult)


def build_market_context() -> Dict:
    """
    main.py から呼ばれる “唯一の市場コンテキスト生成”。
    ImportError の原因になってた関数名はこれで固定：build_market_context
    """
    mkt = calc_market_score()
    mkt_score = int(mkt.get("score", 50))
    delta3d = int(mkt.get("delta3d", 0))

    # events
    # today_date は report 側でも使うが、ここでは risk flag だけ作る
    _, is_risk_day = build_event_warnings()

    # no-trade (機械化)
    nt, nt_reasons = no_trade_rule(mkt_score, delta3d)

    lev, lev_comment = recommend_leverage(mkt_score)
    mult = regime_multiplier(mkt_score, delta3d, is_risk_day)

    mkt["no_trade"] = bool(nt)
    mkt["no_trade_reasons"] = nt_reasons
    mkt["event_risk_day"] = bool(is_risk_day)
    mkt["lev"] = float(lev)
    mkt["lev_comment"] = str(lev_comment)
    mkt["regime_mult"] = float(mult)

    return mkt