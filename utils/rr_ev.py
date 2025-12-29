from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.features import Feat
from utils.util import clamp


@dataclass
class RREV:
    stop: float
    tp1: float
    tp2: float
    rr: float

    pwin: float
    ev_r: float
    adj_ev: float

    expected_days: float
    r_per_day: float

    regime_mult: float


def _swing_low(hist: pd.DataFrame, lookback: int = 12) -> float:
    try:
        low = hist["Low"].astype(float).tail(lookback)
        v = float(low.min())
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def regime_multiplier(market_score: int, delta3d: int, major_event: bool) -> float:
    mult = 1.0
    if market_score >= 60 and delta3d >= 0:
        mult *= 1.05
    if delta3d <= -5:
        mult *= 0.70
    if major_event:
        mult *= 0.75
    return float(clamp(mult, 0.50, 1.10))


def compute_rr_ev(
    hist: pd.DataFrame,
    feat: Feat,
    setup_type: str,
    in_center: float,
    in_low: float,
    market_score: int,
    delta3d: int,
    major_event: bool,
) -> RREV:
    """仕様書v2.0：STOP/TP（2段階） + RR/EV + 速度"""
    atr = feat.atr14
    if not np.isfinite(atr) or atr <= 0:
        atr = max(feat.close * 0.01, 1.0)

    # STOP
    if setup_type == "A":
        stop = in_low - 0.7 * atr  # IN_low - 0.7ATR
        # 直近安値が近い場合（より下へ）
        sl = _swing_low(hist, lookback=12)
        if np.isfinite(sl):
            stop = min(stop, sl - 0.2 * atr)
    elif setup_type == "B":
        stop = in_center - 1.0 * atr
        sl = _swing_low(hist, lookback=12)
        if np.isfinite(sl):
            stop = min(stop, sl - 0.2 * atr)
    else:
        stop = in_low - 0.7 * atr

    # STOPの下限（近すぎ事故を避ける）
    stop = float(min(stop, in_center - 0.3 * atr))
    if stop >= in_center:
        stop = float(in_center - 1.0 * atr)

    risk = in_center - stop
    risk = risk if risk > 0 else max(atr * 0.8, 1.0)

    # TP1/TP2
    tp1 = in_center + 1.5 * risk
    tp2 = in_center + 3.0 * risk

    # 地合いで現実補正（TPは地合い悪いほど控えめ）
    if market_score <= 45 or delta3d <= -5:
        tp1 *= 0.98
        tp2 *= 0.96
    elif market_score >= 70 and delta3d >= 0:
        tp1 *= 1.01
        tp2 *= 1.02

    rr = (tp2 - in_center) / risk if risk > 0 else 0.0
    rr = float(rr)

    # Pwin推定（代理特徴）
    p = 0.30
    # TrendStrength: MA20上&傾き
    if feat.close > feat.ma20 > feat.ma50:
        p += 0.08
    if np.isfinite(feat.ma20_slope_5d) and feat.ma20_slope_5d > 0:
        p += float(clamp(feat.ma20_slope_5d * 10.0, 0.0, 0.06))
    # RSI（過熱なし）
    if 40 <= feat.rsi14 <= 62:
        p += 0.06
    elif 62 < feat.rsi14 <= 70:
        p += 0.02
    else:
        p -= 0.03

    # VolumeQuality（A:押しで枯れ、B:ブレイクで増）
    if setup_type == "A":
        if np.isfinite(feat.volume) and np.isfinite(feat.vol_ma20) and feat.vol_ma20 > 0:
            if feat.volume < feat.vol_ma20:
                p += 0.03
    if setup_type == "B":
        if np.isfinite(feat.volume) and np.isfinite(feat.vol_ma20) and feat.vol_ma20 > 0:
            if feat.volume >= 1.5 * feat.vol_ma20:
                p += 0.04

    # Liquidity（ADV20で微加点）
    adv = feat.turnover_ma20
    if np.isfinite(adv):
        if adv >= 1e9:
            p += 0.03
        elif adv >= 2e8:
            p += 0.015

    # Risk（地合い悪化・イベント）
    if market_score < 50:
        p -= 0.03
    if delta3d <= -5:
        p -= 0.05
    if major_event:
        p -= 0.04

    p = float(clamp(p, 0.15, 0.60))

    ev = p * rr - (1.0 - p) * 1.0
    ev = float(ev)

    mult = regime_multiplier(market_score, delta3d, major_event)
    adj_ev = float(ev * mult)

    # 速度：ExpectedDays = (TP2-IN)/(k*ATR) k=1.0
    expected_days = float((tp2 - in_center) / (1.0 * atr)) if atr > 0 else 999.0
    expected_days = float(clamp(expected_days, 0.5, 30.0))

    r_per_day = float(rr / expected_days) if expected_days > 0 else 0.0

    return RREV(
        stop=float(round(stop, 1)),
        tp1=float(round(tp1, 1)),
        tp2=float(round(tp2, 1)),
        rr=float(rr),
        pwin=float(p),
        ev_r=float(ev),
        adj_ev=float(adj_ev),
        expected_days=float(expected_days),
        r_per_day=float(r_per_day),
        regime_mult=float(mult),
    )