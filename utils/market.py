from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

@dataclass
class MarketContext:
    score: int
    delta_3d: int
    comment: str
    trade_allowed: bool
    no_trade_reasons: list[str]
    regime_multiplier: float
    lev: float
    lev_comment: str
    extras: Dict[str, float]

def _score_from_5d(nk_5d: float, tp_5d: float) -> int:
    base = 50.0
    base += float(np.clip((nk_5d + tp_5d) / 2.0, -20, 20))
    return int(np.clip(round(base), 0, 100))

def _five_day_chg_series(symbol: str, days: int = 20) -> pd.Series:
    try:
        df = yf.Ticker(symbol).history(period=f"{days}d", auto_adjust=True)
        if df is None or df.empty or len(df) < 6:
            return pd.Series(dtype=float)
        c = df["Close"].astype(float)
        # 5営業日変化を各日で計算（簡易：5本前との比）
        chg = (c / c.shift(5) - 1.0) * 100.0
        return chg.dropna()
    except Exception:
        return pd.Series(dtype=float)

def _recent_score_and_delta() -> Tuple[int, int, Dict[str, float]]:
    nk = _five_day_chg_series("^N225", days=25)
    tp = _five_day_chg_series("^TOPX", days=25)
    if nk.empty or tp.empty:
        return 50, 0, {}

    # 同じ日付で揃える
    df = pd.concat([nk.rename("nk"), tp.rename("tp")], axis=1).dropna()
    if len(df) < 8:
        return 50, 0, {}

    # 最終日と3営業日前（=4つ前）で比較
    nk_now = float(df["nk"].iloc[-1])
    tp_now = float(df["tp"].iloc[-1])
    score_now = _score_from_5d(nk_now, tp_now)

    idx_3d = -4 if len(df) >= 4 else -1
    nk_prev = float(df["nk"].iloc[idx_3d])
    tp_prev = float(df["tp"].iloc[idx_3d])
    score_prev = _score_from_5d(nk_prev, tp_prev)

    delta = int(score_now - score_prev)
    extras = {"n225_5d": nk_now, "topix_5d": tp_now}
    return score_now, delta, extras

def _enhance_with_sox_nvda(score: float, extras: Dict[str, float]) -> float:
    # SOX / NVDA を軽く反映（範囲を狭くして暴れないように）
    try:
        sox = yf.Ticker("^SOX").history(period="6d", auto_adjust=True)
        if sox is not None and not sox.empty and len(sox) >= 2:
            chg = float((sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0)
            score += float(np.clip(chg / 2.0, -4.0, 4.0))
            extras["sox_5d"] = chg
    except Exception:
        pass

    try:
        nv = yf.Ticker("NVDA").history(period="6d", auto_adjust=True)
        if nv is not None and not nv.empty and len(nv) >= 2:
            chg = float((nv["Close"].iloc[-1] / nv["Close"].iloc[0] - 1.0) * 100.0)
            score += float(np.clip(chg / 3.0, -3.0, 3.0))
            extras["nvda_5d"] = chg
    except Exception:
        pass

    return score

def _comment(score: int) -> str:
    if score >= 75:
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

def get_market_context() -> MarketContext:
    score0, delta, extras = _recent_score_and_delta()
    score = float(score0)
    score = _enhance_with_sox_nvda(score, extras)
    score = int(np.clip(round(score), 0, 100))

    reasons: list[str] = []
    trade_allowed = True

    # v2.0 NO-TRADE（強制）
    if score < 45:
        trade_allowed = False
        reasons.append("MarketScore<45")
    if delta <= -5 and score < 55:
        trade_allowed = False
        reasons.append("地合い悪化初動(Δ3d<=-5 & Score<55)")

    # RegimeMultiplier
    mult = 1.00
    if score >= 60 and delta >= 0:
        mult = 1.05
    if delta <= -5:
        mult = 0.70

    lev, lev_comment = _recommend_leverage(score)

    return MarketContext(
        score=score,
        delta_3d=int(delta),
        comment=_comment(score),
        trade_allowed=trade_allowed,
        no_trade_reasons=reasons,
        regime_multiplier=float(mult),
        lev=float(lev),
        lev_comment=str(lev_comment),
        extras=extras,
    )
