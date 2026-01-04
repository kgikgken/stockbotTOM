from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from utils.util import clamp


def dynamic_min_rr(mkt_score: int, delta3d: int) -> float:
    """
    (1) RR下限を地合い連動にする
    - 強い相場: 少し緩め
    - 弱い/悪化: 厳しめ
    """
    if mkt_score >= 75:
        base = 1.60
    elif mkt_score >= 65:
        base = 1.70
    elif mkt_score >= 55:
        base = 1.80
    elif mkt_score >= 50:
        base = 1.90
    else:
        base = 2.00

    if delta3d <= -5:
        base += 0.20
    elif delta3d <= -2:
        base += 0.10

    return float(clamp(base, 1.55, 2.30))


def compute_targets(df: pd.DataFrame, setup_type: str, entry: float, stop: float, mkt_score: int) -> Tuple[float, float]:
    """
    TP1/TP2を固定化しない（RR・R/dayが自然に散るように）
    - 抵抗（直近高値）と “欲しいRR” の両方で決める
    """
    close = df["Close"].astype(float)
    if not (np.isfinite(entry) and np.isfinite(stop) and entry > 0 and stop > 0 and entry > stop):
        return entry, entry

    risk = entry - stop

    # 抵抗（60日高値手前）
    window = 60 if len(close) >= 60 else len(close)
    hi = float(close.tail(window).max()) if window >= 5 else float(close.iloc[-1])

    # トレンド強さで “狙うRR” を変える（固定禁止）
    ma20 = float(df["ma20"].iloc[-1]) if "ma20" in df.columns and np.isfinite(df["ma20"].iloc[-1]) else entry
    ma50 = float(df["ma50"].iloc[-1]) if "ma50" in df.columns and np.isfinite(df["ma50"].iloc[-1]) else entry
    trend_strength = 0.0
    if entry > 0 and ma20 > 0 and ma50 > 0:
        if entry > ma20 > ma50:
            trend_strength = 1.0
        elif entry > ma20:
            trend_strength = 0.6
        else:
            trend_strength = 0.3

    # marketで上限/下限を揺らす
    if mkt_score >= 70:
        rr_target = 2.4 + 1.4 * trend_strength  # 2.4〜3.8
    elif mkt_score >= 55:
        rr_target = 2.1 + 1.2 * trend_strength  # 2.1〜3.3
    else:
        rr_target = 2.0 + 0.9 * trend_strength  # 2.0〜2.9

    # setupで微調整（A2は伸び狙い、Bは初動で控えめ）
    if setup_type == "A2":
        rr_target += 0.25
    elif setup_type == "B":
        rr_target -= 0.20

    rr_target = float(clamp(rr_target, 1.80, 4.20))

    # rr_targetベースのtp2
    tp2_by_rr = entry + risk * rr_target

    # 抵抗ベース（抵抗が近いなら現実に合わせる）
    tp2_by_res = min(hi * 0.995, entry + risk * 4.5)

    tp2 = min(tp2_by_rr, tp2_by_res)

    # 最低限、tp2はentryより上
    if tp2 <= entry * 1.01:
        tp2 = entry + risk * 1.8

    # TP1は 1.5R を基準に、tp2の下に収める（固定化しないため小さく揺らす）
    tp1 = entry + risk * (1.35 if setup_type == "B" else 1.50)
    tp1 = min(tp1, entry + (tp2 - entry) * 0.62)

    return float(tp1), float(tp2)


def compute_stop(df: pd.DataFrame, setup_type: str, entry_zone: Dict) -> float:
    """
    Stop：仕様通り + 構造（直近安値）で補強
    """
    entry = float(entry_zone["center"])
    atr = float(entry_zone["atr"])

    low = df["Low"].astype(float)
    lookback = 12 if setup_type != "B" else 8
    swing_low = float(low.tail(lookback).min())

    if setup_type in ("A1", "A2"):
        # STOP = IN_low - 0.7ATR（≒中心 -1.2ATR）
        stop = float(entry_zone["low"] - 0.70 * atr)
        # 構造安値も考慮
        stop = min(stop, swing_low - 0.20 * atr)
    else:
        # B: BreakLine - 1.0ATR（centerをbreakline扱い）
        stop = float(entry - 1.00 * atr)
        stop = min(stop, swing_low - 0.20 * atr)

    # 近すぎ/遠すぎを抑える
    if entry > 0:
        pct = (stop / entry) - 1.0
        pct = clamp(pct, -0.12, -0.02)
        stop = entry * (1.0 + pct)

    return float(stop)


def calc_expected_days(entry: float, tp2: float, atr: float, setup_type: str) -> float:
    """
    ExpectedDays = (tp2-entry)/(k*ATR)
    kを固定しない（R/day分布を広げる）
    """
    if not (np.isfinite(entry) and np.isfinite(tp2) and np.isfinite(atr) and entry > 0 and tp2 > entry and atr > 0):
        return 9.9

    move = tp2 - entry
    atr_pct = atr / entry

    # ATR%が高いほど “進む速度” を上に寄せる（短期向き）
    base_k = 0.90 + clamp((atr_pct - 0.02) * 6.0, -0.10, 0.35)  # 0.80〜1.25くらい

    # setup補正
    if setup_type == "A1":
        base_k *= 1.05
    elif setup_type == "A2":
        base_k *= 0.92
    elif setup_type == "B":
        base_k *= 1.10

    k = clamp(base_k, 0.75, 1.35)

    days = move / (k * atr)
    return float(clamp(days, 1.2, 7.0))


def compute_ev(pwin: float, rr: float) -> float:
    """
    EV(R) = Pwin*RR - (1-Pwin)*1
    """
    if not (np.isfinite(pwin) and np.isfinite(rr) and rr > 0):
        return -999.0
    pwin = clamp(pwin, 0.0, 1.0)
    return float(pwin * rr - (1.0 - pwin) * 1.0)