from __future__ import annotations

import numpy as np
import pandas as pd

from utils.features import atr, sma, rsi, turnover_avg
from utils.util import clamp


def gu_flag(df: pd.DataFrame) -> bool:
    """
    GU判定（概算）:
      Open > PrevClose + 1.0ATR
    """
    if df is None or len(df) < 3:
        return False
    a = atr(df, 14)
    if not np.isfinite(a) or a <= 0:
        return False
    prev_close = float(df["Close"].astype(float).iloc[-2])
    op = float(df["Open"].astype(float).iloc[-1])
    return bool(op > prev_close + 1.0 * a)


def calc_stop_tp(df: pd.DataFrame, setup_type: str, in_center: float, in_low: float) -> tuple[float, float, float]:
    """
    Stop/TP1/TP2 を“仕様書寄せ”で決める
    - Stop:
      A: IN_low - 0.7ATR（≒ center - 1.2ATR）
      B: BreakLine - 1.0ATR
    - TP1: 1.5R, TP2: 3.0R
    """
    c = float(df["Close"].astype(float).iloc[-1])
    a = atr(df, 14)
    if not np.isfinite(a) or a <= 0:
        a = max(c * 0.01, 1.0)

    if setup_type == "B":
        stop = in_center - 1.0 * a
    else:
        stop = in_low - 0.7 * a

    # 直近安値バッファ
    lookback = 12
    swing_low = float(df["Low"].astype(float).tail(lookback).min())
    stop = min(stop, swing_low - 0.2 * a)

    # クランプ（事故防止）
    stop = float(in_center * (1.0 - 0.12)) if stop < in_center * (1.0 - 0.12) else float(stop)
    stop = float(in_center * (1.0 - 0.02)) if stop > in_center * (1.0 - 0.02) else float(stop)

    risk = max(1e-9, in_center - stop)
    tp1 = in_center + 1.5 * risk
    tp2 = in_center + 3.0 * risk
    return float(stop), float(tp1), float(tp2)


def estimate_pwin(df: pd.DataFrame, sector_rank: int | None, adv20: float, gu: bool, mkt_score: int) -> float:
    """
    ログ無しの“代理Pwin”推定（0-1）
    - TrendStrength: MA位置/傾き
    - RSI: 過熱回避
    - Liquidity: ADV20
    - SectorRank: 上位ほど少し加点（選定理由ではなく補助）
    - GU: 強烈減点
    - Market: 地合い
    """
    close = df["Close"].astype(float)
    c = float(close.iloc[-1])
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    r = rsi(close, 14)

    p = 0.35

    if np.isfinite(ma20) and np.isfinite(ma50):
        if c > ma20 > ma50:
            p += 0.10
        elif c > ma20:
            p += 0.05

        # ma20 slope（5日）
        if len(close) >= 26:
            ma20_prev = float(close.rolling(20).mean().iloc[-6])
            if np.isfinite(ma20_prev) and ma20_prev > 0:
                slope = (ma20 - ma20_prev) / ma20_prev
                p += float(clamp(slope * 2.5, -0.05, 0.08))

    if np.isfinite(r):
        if 40 <= r <= 62:
            p += 0.08
        elif r < 35 or r > 75:
            p -= 0.06

    # 流動性（200M以上を基準）
    if np.isfinite(adv20):
        if adv20 >= 1e9:
            p += 0.06
        elif adv20 >= 2e8:
            p += 0.03
        else:
            p -= 0.05

    # セクター補助（上位5以内だけ微加点）
    if sector_rank is not None and 1 <= sector_rank <= 5:
        p += 0.02 * (6 - sector_rank) / 5.0

    # 地合い
    p += float(clamp((mkt_score - 50) * 0.0025, -0.08, 0.10))

    # GU
    if gu:
        p -= 0.25

    return float(clamp(p, 0.05, 0.80))


def calc_ev(rr: float, pwin: float) -> float:
    """
    EV = Pwin*RR - (1-Pwin)*1
    """
    rr = float(rr)
    p = float(pwin)
    return float(p * rr - (1.0 - p) * 1.0)


def regime_multiplier(mkt_score: int, delta3d: float, event_risk: bool) -> float:
    mult = 1.00
    if mkt_score >= 60 and delta3d >= 0:
        mult *= 1.05
    if delta3d <= -5:
        mult *= 0.70
    if event_risk:
        mult *= 0.75
    return float(mult)


def expected_days(tp2: float, entry: float, a: float) -> float:
    """
    ExpectedDays = (TP2-Entry)/(k*ATR), k=1.0固定（ログ無し）
    """
    if not np.isfinite(tp2) or not np.isfinite(entry) or not np.isfinite(a) or a <= 0:
        return 9.9
    return float((tp2 - entry) / (1.0 * a))


def r_per_day(rr: float, exp_days: float) -> float:
    if exp_days <= 0:
        return 0.0
    return float(rr / exp_days)


def adv20_from_df(df: pd.DataFrame) -> float:
    t = turnover_avg(df, 20)
    return float(t) if np.isfinite(t) else float("nan")