# ============================================
# utils/rr_ev.py
# STOP/TP、RR、EV、補正EV、速度（R/day）を作る
# RRを固定化しない（分布を自然にする）
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from utils.features import Features
from utils.util import estimate_days, calc_r_per_day, clamp


@dataclass
class RREV:
    stop: float
    tp1: float
    tp2: float
    rr: float
    ev: float
    adj_ev: float
    expected_days: float
    r_per_day: float


def _pwin_proxy(f: Features, setup_type: str) -> float:
    """
    勝率の代理（0-1）
    """
    p = 0.50

    # トレンド強度（sma構造）
    if f.close > f.sma20 > f.sma50:
        p += 0.08
    if (f.sma20 - f.sma50) > 0:
        p += 0.04

    # RSI（過熱なら下げる）
    if f.rsi14 >= 70:
        p -= 0.08
    elif 45 <= f.rsi14 <= 62:
        p += 0.05
    elif f.rsi14 < 35:
        p -= 0.06

    # A1/A2の差
    if setup_type == "A1":
        p += 0.03
    if setup_type == "A2":
        p -= 0.01

    # ATR%（動かなすぎは不利、荒すぎも不利）
    if f.atrp14 < 1.5:
        p -= 0.10
    elif f.atrp14 > 6.0:
        p -= 0.05
    else:
        p += 0.02

    # GUは大幅減点（本来は候補外）
    if f.gu_flag:
        p -= 0.20

    return clamp(p, 0.05, 0.85)


def build_rr_ev(
    f: Features,
    setup_type: str,
    market_score: float,
    macro_risk: bool,
) -> Optional[RREV]:
    """
    STOP/TPは構造から出す（RR固定化しない）
    - STOP: entry_low - 0.7ATR（A1） / -0.9ATR（A2）
    - TP2: 過去20日高値やATRから伸び代を見て可変
    - TP1: TP2の手前（部分利確）
    """
    atr = f.atr14
    if atr <= 0:
        return None

    # entryは別で算出しているが、ここではSMA20基準（entry.pyと整合）
    entry = f.sma20

    # STOP
    if setup_type == "A1":
        stop = (entry - 1.2 * atr)
    elif setup_type == "A2":
        stop = (entry - 1.4 * atr)
    else:
        stop = (entry - 1.2 * atr)

    # 伸び代（RR分布を自然にする要）
    # 地合いが強いほど伸び代を少し取りやすい
    # ボラが高すぎるものは伸び代控えめ
    base_mult = 2.0
    if market_score >= 70:
        base_mult = 2.4
    elif market_score >= 60:
        base_mult = 2.2
    elif market_score >= 50:
        base_mult = 2.0
    else:
        base_mult = 1.9

    if f.atrp14 > 6.0:
        base_mult -= 0.2

    # A1は伸びやすい
    if setup_type == "A1":
        base_mult += 0.2
    if setup_type == "A2":
        base_mult -= 0.1

    # マクロ警戒は控えめ（RRを上げるのではなく、補正EVで抑える）
    if macro_risk:
        base_mult -= 0.1

    # TP2
    tp2 = entry + base_mult * (entry - stop)

    # TP1はTP2の手前（部分利確）
    tp1 = entry + 0.6 * (tp2 - entry)

    # RR
    risk = (entry - stop)
    if risk <= 0:
        return None
    rr = (tp2 - entry) / risk

    # Pwin推定 → EV
    pwin = _pwin_proxy(f, setup_type)
    ev = pwin * rr - (1.0 - pwin) * 1.0

    # 地合い補正（現実値へ）
    mult = 1.0
    if market_score >= 70:
        mult *= 1.05
    elif market_score >= 60:
        mult *= 1.00
    elif market_score >= 50:
        mult *= 0.90
    else:
        mult *= 0.75

    # イベント接近（マクロ警戒）
    if macro_risk:
        mult *= 0.75

    adj_ev = ev * mult

    # 速度
    days = estimate_days(tp2, entry, atr)
    # 速度分布を広げる：ATRだけでなく「伸び代」を織り込む係数
    # 伸び代大(=RR高)ほど日数増えやすいが、一定上は抑える
    days = days * clamp(0.85 + (rr - 2.0) * 0.10, 0.75, 1.25)

    r_day = calc_r_per_day(rr, days)

    return RREV(
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        rr=rr,
        ev=ev,
        adj_ev=adj_ev,
        expected_days=days,
        r_per_day=r_day,
    )