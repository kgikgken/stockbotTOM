# utils/market.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import yfinance as yf


# ============================================================
# Market symbols (JP)
# ============================================================
NIKKEI = "^N225"
TOPIX = "^TOPX"


@dataclass(frozen=True)
class MarketState:
    score: int
    comment: str
    delta_3d: int
    n225_5d: float
    topix_5d: float


def _fetch_close(symbol: str, period: str = "260d") -> np.ndarray:
    """
    close array (float). empty -> []
    """
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return np.array([], dtype=float)
        arr = df["Close"].astype(float).to_numpy()
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([], dtype=float)


def _sma(arr: np.ndarray, window: int) -> float:
    if arr.size < window:
        return float(arr[-1]) if arr.size else float("nan")
    return float(np.mean(arr[-window:]))


def _rsi(arr: np.ndarray, period: int = 14) -> float:
    if arr.size < period + 2:
        return float("nan")
    diff = np.diff(arr)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:]) + 1e-12
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _pct_change(arr: np.ndarray, n: int) -> float:
    if arr.size < n + 1:
        return 0.0
    a0 = float(arr[-(n + 1)])
    a1 = float(arr[-1])
    if not np.isfinite(a0) or not np.isfinite(a1) or a0 <= 0:
        return 0.0
    return float((a1 / a0 - 1.0) * 100.0)


def _drawdown_20d(arr: np.ndarray) -> float:
    """
    直近20日高値からの下落率（%）
    """
    if arr.size < 21:
        return 0.0
    win = arr[-20:]
    hi = float(np.max(win))
    last = float(win[-1])
    if hi <= 0 or not np.isfinite(hi) or not np.isfinite(last):
        return 0.0
    return float((last / hi - 1.0) * 100.0)  # negative when down


def _market_comment(score: int) -> str:
    if score >= 75:
        return "強め"
    if score >= 65:
        return "やや強め"
    if score >= 55:
        return "中立"
    if score >= 45:
        return "弱め"
    return "弱い"


def calc_market_score() -> Dict:
    """
    World-Class v2系：指数の「トレンド + 勢い + リスク」を合成して 0-100
    出力：
      score, comment, delta_3d, n225_5d, topix_5d
    """
    nk = _fetch_close(NIKKEI, "260d")
    tp = _fetch_close(TOPIX, "260d")

    # fallback
    if nk.size < 60 and tp.size < 60:
        score = 50
        return {"score": score, "comment": _market_comment(score), "delta_3d": 0, "n225_5d": 0.0, "topix_5d": 0.0}

    # features (平均化して安定化)
    def feat(arr: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        ma20 = _sma(arr, 20)
        ma50 = _sma(arr, 50)
        ma10 = _sma(arr, 10)
        last = float(arr[-1]) if arr.size else float("nan")
        rsi14 = _rsi(arr, 14)
        chg5 = _pct_change(arr, 5)
        chg20 = _pct_change(arr, 20)
        dd20 = _drawdown_20d(arr)
        # trend flags
        trend1 = 1.0 if (np.isfinite(last) and np.isfinite(ma50) and last > ma50) else 0.0
        trend2 = 1.0 if (np.isfinite(ma20) and np.isfinite(ma50) and ma20 > ma50) else 0.0
        # momentum
        mom = 0.5 * chg5 + 0.5 * chg20
        # risk: under ma10 / drawdown
        under_ma10 = 1.0 if (np.isfinite(last) and np.isfinite(ma10) and last < ma10) else 0.0
        return trend1, trend2, mom, rsi14, dd20, under_ma10

    nk_t1, nk_t2, nk_mom, nk_rsi, nk_dd, nk_under10 = feat(nk)
    tp_t1, tp_t2, tp_mom, tp_rsi, tp_dd, tp_under10 = feat(tp)

    # blend
    trend = (nk_t1 + nk_t2 + tp_t1 + tp_t2) / 4.0  # 0..1
    mom = (nk_mom + tp_mom) / 2.0                  # %
    rsi = (nk_rsi + tp_rsi) / 2.0                  # 0..100
    dd = (nk_dd + tp_dd) / 2.0                     # negative
    under10 = (nk_under10 + tp_under10) / 2.0      # 0..1

    # score build
    score = 50.0

    # trend contributes up to +18
    score += 18.0 * trend

    # momentum contributes roughly +-18 (clamped)
    score += float(np.clip(mom, -6.0, 6.0)) * 3.0

    # RSI (avoid overheat; 45-65 best)
    if np.isfinite(rsi):
        if 45 <= rsi <= 65:
            score += 8.0
        elif 35 <= rsi < 45 or 65 < rsi <= 75:
            score += 4.0
        else:
            score += 0.0

    # risk penalty: under MA10 and drawdown
    score -= 10.0 * under10
    if np.isfinite(dd):
        # if drawdown worse than -4%, penalize progressively
        if dd <= -4.0:
            score -= float(np.clip((-dd - 4.0), 0.0, 8.0)) * 1.2

    score = float(np.clip(score, 0.0, 100.0))
    score_i = int(round(score))

    # delta_3d : score change proxy from 3d momentum (integer)
    # 3d momentum roughly (avg chg3) mapped to -20..+20
    nk3 = _pct_change(nk, 3)
    tp3 = _pct_change(tp, 3)
    chg3 = (nk3 + tp3) / 2.0
    delta_3d = int(round(np.clip(chg3 * 4.0, -20.0, 20.0)))  # 0.25% => 1

    out = {
        "score": score_i,
        "comment": _market_comment(score_i),
        "delta_3d": delta_3d,
        "n225_5d": float(_pct_change(nk, 5)),
        "topix_5d": float(_pct_change(tp, 5)),
    }
    return out


def enhance_market_score() -> Dict:
    """
    互換：main.py が呼ぶ想定。
    calc_market_score を基準に、外部の影響（SOX/NVDA）は *軽く* だけ反映。
    """
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX（米半導体）: 強すぎる反映は禁止（日本株全体を壊す）
    try:
        sox = _fetch_close("^SOX", "30d")
        if sox.size >= 6:
            sox5 = _pct_change(sox, 5)
            score += float(np.clip(sox5 / 2.5, -4.0, 4.0))
            mkt["sox_5d"] = float(sox5)
    except Exception:
        pass

    # NVDA（代表）
    try:
        nv = _fetch_close("NVDA", "30d")
        if nv.size >= 6:
            nv5 = _pct_change(nv, 5)
            score += float(np.clip(nv5 / 3.5, -3.0, 3.0))
            mkt["nvda_5d"] = float(nv5)
    except Exception:
        pass

    score_i = int(round(np.clip(score, 0.0, 100.0)))
    mkt["score"] = score_i
    mkt["comment"] = _market_comment(score_i)
    return mkt