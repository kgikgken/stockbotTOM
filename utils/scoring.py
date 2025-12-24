from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.rr import TradePlan


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float("nan")


def _sma(s: pd.Series, w: int) -> float:
    if s is None or len(s) < w:
        return _last(s)
    v = s.rolling(w).mean().iloc[-1]
    return float(v) if np.isfinite(v) else _last(s)


def _rsi(close: pd.Series, period: int = 14) -> float:
    if close is None or len(close) < period + 2:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def _clamp01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _z_to_01(z: float, z_min: float, z_max: float) -> float:
    if not np.isfinite(z):
        return 0.0
    if z_max <= z_min:
        return 0.0
    return float(np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0))


def passes_universe_filters(
    hist: pd.DataFrame,
    price_min: float,
    price_max: float,
    adv_min: float,
    atrp_min: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    Universeフィルタ（必須）: 価格 / 流動性(ADV20売買代金) / ボラ(ATR%)
    """
    if hist is None or len(hist) < 120:
        return False, {}

    df = hist.copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    c = _last(close)
    if not np.isfinite(c) or not (price_min <= c <= price_max):
        return False, {}

    # ADV20（売買代金）
    adv20 = float((close * vol).rolling(20).mean().iloc[-1]) if len(close) >= 20 else float("nan")
    if not np.isfinite(adv20) or adv20 < adv_min:
        return False, {}

    # ATR%(14)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float("nan")
    atrp = float(atr14 / c) if np.isfinite(atr14) and np.isfinite(c) and c > 0 else float("nan")
    if not np.isfinite(atrp) or atrp < atrp_min:
        return False, {}

    return True, {"adv20": float(adv20), "atr14": float(atr14), "atrp": float(atrp)}


def estimate_pwin(
    hist: pd.DataFrame,
    plan: TradePlan,
    sector_rank01: float,
    adv20: float,
    market_score: int,
) -> float:
    """
    Pwin 推定（代理特徴の合成）。
    目的は「EVで殺し切る」ための相対評価。絶対勝率の当てではない。
    """
    df = hist.copy()
    close = df["Close"].astype(float)
    c = _last(close)

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    rsi = _rsi(close, 14)

    # TrendStrength
    trend = 0.0
    if np.isfinite(sma20) and np.isfinite(sma50) and sma20 > sma50:
        trend += 0.5
    if np.isfinite(c) and np.isfinite(sma20) and c > sma20:
        trend += 0.5
    trend = _clamp01(trend)

    # RSI quality
    if not np.isfinite(rsi):
        rsi_q = 0.4
    elif 40 <= rsi <= 60:
        rsi_q = 1.0
    elif 35 <= rsi < 40 or 60 < rsi <= 65:
        rsi_q = 0.6
    else:
        rsi_q = 0.2

    # RS proxy (20d return)
    rs = 0.0
    if len(close) >= 21:
        rs20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        rs = _z_to_01(rs20, -0.10, 0.20)

    # VolumeQuality
    volq = 0.5
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        v_now = _last(vol)
        v_ma20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float("nan")
        if np.isfinite(v_now) and np.isfinite(v_ma20) and v_ma20 > 0:
            if plan.setup == "B":
                volq = _z_to_01(v_now / v_ma20, 1.0, 2.5)
            else:
                volq = _z_to_01(v_ma20 / max(v_now, 1.0), 0.8, 1.6)

    # Liquidity
    liq = _z_to_01(adv20, 1e8, 1e9)

    # GapRisk
    gap = 1.0
    if plan.gu_flag:
        gap = 0.0
    elif plan.in_distance_atr > 0.8:
        gap = 0.2
    elif plan.action == "指値待ち":
        gap = 0.6

    # Market
    mkt = _z_to_01(float(market_score), 45.0, 70.0)

    sec = _clamp01(sector_rank01)

    p = (
        0.22 * trend
        + 0.16 * rsi_q
        + 0.18 * rs
        + 0.14 * volq
        + 0.10 * liq
        + 0.10 * sec
        + 0.10 * mkt
    )

    # Gap は最後に強制的に効かせる
    p *= (0.55 + 0.45 * gap)

    return float(np.clip(p, 0.15, 0.70))


def compute_ev(pwin: float, r: float) -> float:
    if not np.isfinite(pwin) or not np.isfinite(r) or r <= 0:
        return -999.0
    return float(pwin * r - (1.0 - pwin) * 1.0)


def regime_multiplier(market_score: int, d_market_3d: int, event_penalty: float) -> float:
    mult = 1.0
    if market_score >= 60 and d_market_3d >= 0:
        mult *= 1.05
    if d_market_3d <= -5:
        mult *= 0.70
    mult *= float(event_penalty)
    return float(mult)
