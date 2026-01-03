　# utils/rr_ev.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ----------------------------
# ユーティリティ
# ----------------------------
def _safe(v, default=np.nan) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


# ----------------------------
# ATR（14）
# ----------------------------
def atr14(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)

    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return _safe(v)


def atr_pct(df: pd.DataFrame) -> float:
    a = atr14(df)
    c = _safe(df["Close"].iloc[-1])
    if not np.isfinite(a) or not np.isfinite(c) or c <= 0:
        return np.nan
    return float(a / c * 100.0)


# ----------------------------
# Stop / Target 設計（仕様寄せ）
# ----------------------------
def calc_stop_tp(
    df: pd.DataFrame,
    *,
    setup_type: str,
    entry: float,
    in_low: float,
    in_high: float,
    atr: float,
) -> Dict[str, float]:
    """
    仕様（v2.0）：
      Stop:
        A: STOP = IN_low - 0.7ATR（≒ in_center - 1.2ATR）
        B: STOP = BreakLine(HH20) - 1.0ATR
        直近安値が近い場合: STOP = min(上記, SwingLow - buffer)

      Target:
        TP1 = IN + 1.5R
        TP2 = IN + 3.0R
    """
    if atr <= 0 or not np.isfinite(atr):
        atr = max(entry * 0.01, 1.0)

    # 構造（直近安値）
    lookback = 12
    swing_low = _safe(df["Low"].astype(float).tail(lookback).min(), entry * 0.9)

    buffer = 0.2 * atr

    if setup_type == "A":
        stop_base = in_low - 0.7 * atr
    elif setup_type == "B":
        # Bは entry がHH20近傍の想定
        stop_base = entry - 1.0 * atr
    else:
        stop_base = entry - 1.0 * atr

    stop_struct = swing_low - buffer
    stop = min(stop_base, stop_struct)

    # stopが近すぎ/遠すぎをクランプ（事故防止）
    # entry基準で -2%〜-10%
    sl_pct = (stop / entry) - 1.0
    sl_pct = _clip(sl_pct, -0.10, -0.02)
    stop = entry * (1.0 + sl_pct)

    # R（1R幅）
    one_r = entry - stop
    if one_r <= 0:
        one_r = max(0.5 * atr, 1.0)

    tp1 = entry + 1.5 * one_r
    tp2 = entry + 3.0 * one_r

    return {
        "stop": float(round(stop, 1)),
        "tp1": float(round(tp1, 1)),
        "tp2": float(round(tp2, 1)),
        "one_r": float(one_r),
        "sl_pct": float(sl_pct),
        "tp2_pct": float((tp2 / entry) - 1.0) if entry > 0 else 0.0,
    }


def calc_rr(entry: float, stop: float, tp2: float) -> float:
    if entry <= 0:
        return 0.0
    risk = entry - stop
    reward = tp2 - entry
    if risk <= 0:
        return 0.0
    return float(reward / risk)


# ----------------------------
# Pwin（代理特徴で推定：ログ無しで現実寄せ）
# ----------------------------
def estimate_pwin_proxy(
    df: pd.DataFrame,
    *,
    setup_type: str,
    sector_rank: int | None,
    adv20: float,
    gu_flag: bool,
) -> float:
    """
    0〜1
    代理特徴：
      - TrendStrength（MA構造・傾き）
      - RSI（過熱回避）
      - Liquidity（ADV）
      - SectorRank（上位ほど）
      - GapRisk（GUは強烈減点）
    """
    c = df["Close"].astype(float)
    if len(c) < 60:
        return 0.40

    sma20 = _safe(c.rolling(20).mean().iloc[-1])
    sma50 = _safe(c.rolling(50).mean().iloc[-1])
    sma20_prev = _safe(c.rolling(20).mean().iloc[-6])  # 5営業日前
    trend_slope = (sma20 / sma20_prev - 1.0) if (np.isfinite(sma20) and np.isfinite(sma20_prev) and sma20_prev > 0) else 0.0

    # RSI14
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]
    rs = _safe(avg_gain) / (_safe(avg_loss) + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi = _safe(rsi, 50.0)

    p = 0.42

    # Setup優遇（A > B）
    if setup_type == "A":
        p += 0.05
    elif setup_type == "B":
        p += 0.02

    # MA構造・傾き
    if np.isfinite(sma20) and np.isfinite(sma50):
        if sma20 > sma50:
            p += 0.04
    p += _clip(trend_slope * 4.0, -0.03, 0.06)

    # RSI（過熱は落とす / 押し目は少し上げる）
    if 40 <= rsi <= 62:
        p += 0.05
    elif rsi < 35:
        p -= 0.03
    elif rsi > 70:
        p -= 0.06

    # 流動性（ADV20）
    # 200Mで基準、1Bで上限
    if np.isfinite(adv20):
        if adv20 >= 1e9:
            p += 0.06
        elif adv20 >= 2e8:
            p += 0.03 + 0.03 * (adv20 - 2e8) / (8e8)

    # セクター順位（1位=+、5位以内=+、圏外=0）
    if sector_rank is not None and sector_rank > 0:
        if sector_rank <= 5:
            p += 0.05 * (6 - sector_rank) / 5.0

    # GUは即死級に落とす
    if gu_flag:
        p -= 0.12

    return float(_clip(p, 0.20, 0.70))


# ----------------------------
# EV / AdjEV（地合いで現実値に補正）
# ----------------------------
def regime_multiplier(
    *,
    mkt_score: int,
    delta3d: int,
    is_event_risk: bool,
) -> float:
    """
    仕様例：
      MarketScore>=60 & Δ3d>=0 -> 1.05
      Δ3d<=-5 -> 0.70
      event前日 -> 0.75
    """
    mult = 1.0

    if mkt_score >= 60 and delta3d >= 0:
        mult *= 1.05
    if delta3d <= -5:
        mult *= 0.70
    if is_event_risk:
        mult *= 0.75

    # クリップ（過剰に上下させない）
    return float(_clip(mult, 0.50, 1.15))


def calc_ev(pwin: float, rr: float) -> float:
    # EV = p*R - (1-p)*1
    return float(pwin * rr - (1.0 - pwin) * 1.0)


# ----------------------------
# 速度（ExpectedDays / Rday）
# ----------------------------
def expected_days(
    *,
    entry: float,
    tp2: float,
    atr: float,
    k: float = 1.0,
) -> float:
    """
    ExpectedDays = (TP2 - IN) / (k*ATR)
    kは0.8〜1.2を想定、ここは1.0固定でOK（後で調整可）
    """
    if atr <= 0 or not np.isfinite(atr):
        return 99.0
    move = tp2 - entry
    if move <= 0:
        return 99.0
    return float(move / (k * atr))


def r_per_day(rr: float, exp_days: float) -> float:
    if exp_days <= 0 or not np.isfinite(exp_days):
        return 0.0
    return float(rr / exp_days)


# ----------------------------
# 統合：RR/EV/AdjEV/Rday を算出して返す
# ----------------------------
def compute_rr_ev_bundle(
    df: pd.DataFrame,
    *,
    setup_type: str,
    entry: float,
    in_low: float,
    in_high: float,
    atr: float,
    mkt_score: int,
    delta3d: int,
    is_event_risk: bool,
    sector_rank: int | None,
    adv20: float,
    gu_flag: bool,
) -> Dict[str, float]:
    st = calc_stop_tp(
        df,
        setup_type=setup_type,
        entry=entry,
        in_low=in_low,
        in_high=in_high,
        atr=atr,
    )

    stop = float(st["stop"])
    tp1 = float(st["tp1"])
    tp2 = float(st["tp2"])
    rr = calc_rr(entry, stop, tp2)

    pwin = estimate_pwin_proxy(
        df,
        setup_type=setup_type,
        sector_rank=sector_rank,
        adv20=adv20,
        gu_flag=gu_flag,
    )

    ev = calc_ev(pwin, rr)
    mult = regime_multiplier(
        mkt_score=mkt_score,
        delta3d=delta3d,
        is_event_risk=is_event_risk,
    )
    adj_ev = ev * mult

    exp_days = expected_days(entry=entry, tp2=tp2, atr=atr, k=1.0)
    rday = r_per_day(rr, exp_days)

    return {
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr": float(rr),
        "pwin": float(pwin),
        "ev": float(ev),
        "adj_ev": float(adj_ev),
        "regime_mult": float(mult),
        "expected_days": float(exp_days),
        "r_per_day": float(rday),
        "sl_pct": float(st["sl_pct"]),
        "tp2_pct": float(st["tp2_pct"]),
    }