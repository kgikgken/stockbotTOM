from __future__ import annotations

import numpy as np
import pandas as pd

from utils.util import clamp


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev = close.shift(1)

    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else np.nan


def rr_from_structure(hist: pd.DataFrame, setup: str, in_center: float, mkt_score: int) -> dict:
    """
    RR固定化排除（自然分布）：
    - STOP：構造（直近安値）+ ATR buffer
    - TP2：抵抗帯（60d高値）手前 or 伸び余地（地合いで微調整）
    - ここで丸め/クリップして “3.00固定” にしない
    """
    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)

    price_now = float(close.iloc[-1])
    atr = _atr(hist, 14)
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(in_center) and in_center > 0):
        return {"rr": 0.0, "stop": np.nan, "tp1": np.nan, "tp2": np.nan, "atr": np.nan, "expected_days": np.nan}

    # STOP
    lookback = 12 if setup == "A" else 20
    swing_low = float(low.tail(lookback).min())
    # Aは浅い押し目想定：in_center - 1.2ATR を基準、構造安値が近いならそっち
    if setup == "A":
        stop = min(in_center - 1.2 * atr, swing_low - 0.2 * atr)
    else:
        # Bはブレイク：ブレイクライン割れを許容しない
        stop = min(in_center - 1.0 * atr, swing_low - 0.2 * atr)

    # stopが近すぎる/遠すぎるの事故防止（“固定RR”ではなく“安全レンジ”）
    stop = float(clamp(stop, in_center * 0.90, in_center * 0.98))

    r = in_center - stop
    if not (np.isfinite(r) and r > 0):
        return {"rr": 0.0, "stop": stop, "tp1": np.nan, "tp2": np.nan, "atr": atr, "expected_days": np.nan}

    # TP2（抵抗帯）
    hi_window = 60 if len(close) >= 60 else len(close)
    res = float(close.tail(hi_window).max())

    # 伸び余地：地合いで微調整（弱いと届かない前提で控えめ）
    stretch = 1.00
    if mkt_score >= 70:
        stretch = 1.05
    elif mkt_score <= 45:
        stretch = 0.92

    # “理想TP2”= in + kR（kは固定にしない。抵抗までの距離で自動決定）
    # resまで近い→RR小さくなる、遠い→RR大きくなる（自然分布）
    tp2_cap = res * 0.99
    tp2_raw = in_center + (tp2_cap - in_center) * stretch
    # ただし最低限の伸びは必要（極端な低RR回避）
    tp2_min = in_center + 1.8 * r  # RR>=1.8最低ライン
    tp2 = max(tp2_min, tp2_raw)

    rr = (tp2 - in_center) / r

    # TP1は部分利確（1.5R固定はOK。ここは運用ルール）
    tp1 = in_center + 1.5 * r

    # ExpectedDays（ATRで距離割り）
    expected_days = (tp2 - in_center) / (0.9 * atr)
    expected_days = float(clamp(expected_days, 1.0, 7.0))

    return {
        "rr": float(rr),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "atr": float(atr),
        "expected_days": float(expected_days),
    }


def pwin_proxy(setup: str, sector_rank: int, adv20: float, rsi: float, deviation: float) -> float:
    """
    ログ無しでの勝率代理（固定値禁止）：
    - セクター強いほど +、流動性 +、RSI過熱は -、乖離大は -
    """
    p = 0.26

    # setup補正（A優遇）
    if setup == "A":
        p += 0.06
    else:
        p += 0.03

    # sector_rank（1が最強）
    if sector_rank > 0:
        p += float(clamp((10 - sector_rank) / 50.0, -0.05, 0.12))

    # 流動性
    if np.isfinite(adv20):
        if adv20 >= 1_000_000_000:
            p += 0.08
        elif adv20 >= 200_000_000:
            p += 0.04

    # RSI過熱抑制
    if np.isfinite(rsi):
        if rsi >= 70:
            p -= 0.10
        elif rsi >= 62:
            p -= 0.05
        elif 40 <= rsi <= 62:
            p += 0.03

    # 乖離（追いかけ禁止）
    if np.isfinite(deviation):
        if deviation > 0.8:
            p -= 0.10
        elif deviation > 0.6:
            p -= 0.05
        else:
            p += 0.02

    return float(clamp(p, 0.15, 0.60))


def calc_ev(rr: float, pwin: float) -> float:
    """
    EV = Pwin*RR - (1-Pwin)*1
    """
    return float(pwin * rr - (1.0 - pwin) * 1.0)