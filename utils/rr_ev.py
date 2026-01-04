# utils/rr_ev.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from utils.util import MarketState, clamp


def rr_min_by_market(market_score: float) -> float:
    """
    地合いが弱いほど最低RRを上げる（弱い日に低RRを取らない）
    """
    if market_score >= 70:
        return 1.8
    if market_score >= 60:
        return 2.0
    if market_score >= 50:
        return 2.2
    if market_score >= 45:
        return 2.4
    return 9.9  # 実質禁止


def regime_multiplier(market: MarketState, macro_event_near: bool) -> float:
    mult = 1.0
    if market.delta_3d <= -5:
        mult *= 0.70
    elif market.score >= 60 and market.delta_3d >= 0:
        mult *= 1.05

    if macro_event_near:
        mult *= 0.75

    return float(mult)


def expected_days(tp2: float, entry: float, atr: float, ret20: float, atr_pct: float) -> float:
    """
    R/日分布を広げるため、kを固定にしない。
    - 20日上昇が強いほど（ret20↑）到達が早い想定（days↓）
    - ATR%が高いほど動きやすい（days↓）
    """
    if atr <= 0:
        return 9.9

    dist = max(tp2 - entry, 0.0)
    # k（1日あたりに進むATRの期待値）を可変化
    k = 0.85
    k += clamp(ret20 * 3.0, -0.20, 0.35)         # 勢い補正
    k += clamp((atr_pct - 0.02) * 2.0, -0.15, 0.25)  # ボラ補正
    k = clamp(k, 0.65, 1.25)

    days = dist / (k * atr)
    return float(clamp(days, 1.8, 7.0))


def pwin_proxy(trend_strength: float, rs: float, volume_quality: float, liquidity: float, gap_risk: float) -> float:
    """
    代理Pwin（0〜1）。シンプルで良い。過学習しない。
    """
    x = 0.0
    x += 0.35 * trend_strength
    x += 0.25 * rs
    x += 0.20 * volume_quality
    x += 0.10 * liquidity
    x -= 0.30 * gap_risk
    return float(clamp(0.45 + x, 0.05, 0.90))


def ev(pwin: float, rr: float) -> float:
    """
    EV = Pwin*RR - (1-Pwin)*1
    """
    return float(pwin * rr - (1.0 - pwin) * 1.0)


@dataclass(frozen=True)
class RRResult:
    stop: float
    tp1: float
    tp2: float
    rr: float
    ev: float
    adj_ev: float
    exp_days: float
    r_per_day: float


def compute_rr_ev(
    entry: float,
    in_low: float,
    atr: float,
    market: MarketState,
    macro_event_near: bool,
    trend_strength: float,
    rs: float,
    volume_quality: float,
    liquidity: float,
    gap_risk: float,
    ret20: float,
    atr_pct: float,
) -> RRResult:
    # Stop（仕様の近似）
    stop = in_low - 0.7 * atr
    risk = max(entry - stop, 1e-6)

    # TP：RRを固定しない（自然な分布）
    # まずRRの「狙い値」を、勢いと地合いで少し動かす
    base_rr = 2.4
    base_rr += clamp(ret20 * 4.0, -0.4, 0.8)        # 勢いで上振れ
    base_rr += clamp((market.score - 55) / 100, -0.2, 0.3)
    base_rr = clamp(base_rr, 1.8, 3.8)

    tp2 = entry + base_rr * risk
    tp1 = entry + 1.5 * risk

    rr = (tp2 - entry) / risk

    # Pwin proxy
    pw = pwin_proxy(trend_strength, rs, volume_quality, liquidity, gap_risk)
    raw_ev = ev(pw, rr)

    # 環境補正
    mult = regime_multiplier(market, macro_event_near)
    adj_ev = raw_ev * mult

    # 速度
    exp_days = expected_days(tp2=tp2, entry=entry, atr=atr, ret20=ret20, atr_pct=atr_pct)
    rpd = rr / exp_days if exp_days > 0 else 0.0

    return RRResult(
        stop=float(stop),
        tp1=float(tp1),
        tp2=float(tp2),
        rr=float(rr),
        ev=float(raw_ev),
        adj_ev=float(adj_ev),
        exp_days=float(exp_days),
        r_per_day=float(rpd),
    )