from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from utils.features import FeaturePack
from utils.entry import EntryPlan
from utils.market import MarketContext
from utils.events import EventContext

@dataclass
class EvalPack:
    stop: float
    tp1: float
    tp2: float
    rr: float
    expected_days: float
    r_per_day: float
    pwin: float
    ev: float
    adj_ev: float
    reject_reason: str  # "" if ok

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _pwin_proxy(feat: FeaturePack, sector_rank: int, entry: EntryPlan) -> float:
    # 0〜1：代理特徴で合成（ログ無しでも暴れにくく）
    # TrendStrength
    trend = 0.0
    if feat.close > feat.sma20 > feat.sma50:
        trend += 0.6
    if feat.sma20_slope_5d > 0.0:
        trend += 0.4
    trend = _clip01(trend)

    # RS（-0.05〜+0.05 くらいに収める想定）
    rs = _clip01(0.5 + (feat.rs_20d / 0.10))

    # Sector rank（上位ほど）
    if sector_rank <= 0:
        sec = 0.4
    else:
        sec = _clip01(1.0 - (sector_rank - 1) * 0.15)  # rank1=1.0, rank5=0.4
        sec = max(sec, 0.2)

    # VolumeQuality
    volq = 0.5
    if np.isfinite(feat.vol) and np.isfinite(feat.vol_ma20) and feat.vol_ma20 > 0:
        if feat.vol >= 1.5 * feat.vol_ma20:
            volq = 0.8
        elif feat.vol <= 0.9 * feat.vol_ma20:
            volq = 0.6
    volq = _clip01(volq)

    # GapRisk
    gap = 0.0 if entry.gu_flag else 1.0

    # Liquidity（ADV20）
    liq = 0.0
    if np.isfinite(feat.adv20):
        if feat.adv20 >= 1_000_000_000:
            liq = 1.0
        elif feat.adv20 >= 200_000_000:
            liq = 0.7
        elif feat.adv20 >= 100_000_000:
            liq = 0.5
        else:
            liq = 0.2

    # 合成（勝率は捨てるが、EV計算用の0.35〜0.60帯を中心に）
    raw = (
        0.28 * trend +
        0.22 * rs +
        0.18 * sec +
        0.14 * volq +
        0.10 * gap +
        0.08 * liq
    )
    # 下限/上限を絞る（ログ無しで暴れない）
    return float(np.clip(raw, 0.28, 0.62))

def evaluate(setup_type: str, feat: FeaturePack, entry: EntryPlan, sector_rank: int,
             mkt: MarketContext, ev_ctx: EventContext,
             k_atr: float = 1.0) -> EvalPack:
    atr = feat.atr14 if np.isfinite(feat.atr14) and feat.atr14 > 0 else max(feat.close * 0.01, 1.0)

    # Stop
    if setup_type == "A":
        stop = entry.in_low - 0.7 * atr
    else:
        stop = entry.in_center - 1.0 * atr

    # 直近安値で下に置く（保守的）
    swing_low = feat.low  # last low as fallback
    stop = min(stop, swing_low - 0.2 * atr)

    in_price = entry.in_center  # 評価基準のINは中心（帯で指値しても中心近辺）
    if not (np.isfinite(in_price) and np.isfinite(stop)) or in_price <= stop:
        return EvalPack(0,0,0,0,0,0,0,0,0,"STOP不正")

    # TP2= IN + 3R、TP1= IN +1.5R
    r_unit = in_price - stop
    tp1 = in_price + 1.5 * r_unit
    tp2 = in_price + 3.0 * r_unit

    rr = (tp2 - in_price) / r_unit  # 理論上3.0固定
    rr = float(rr)

    # RR足切り（仕様：R>=2.2）
    if rr < 2.2:
        return EvalPack(stop,tp1,tp2,rr,0,0,0,0,0,"RR不足")

    # ExpectedDays
    expected_days = float((tp2 - in_price) / (k_atr * atr)) if atr > 0 else 999.0
    if not np.isfinite(expected_days) or expected_days <= 0:
        return EvalPack(stop,tp1,tp2,rr,0,0,0,0,0,"ExpectedDays不正")
    if expected_days > 5.0:
        return EvalPack(stop,tp1,tp2,rr,expected_days,0,0,0,0,"遅い(>5日)")

    r_per_day = float(rr / expected_days)
    if r_per_day < 0.5:
        return EvalPack(stop,tp1,tp2,rr,expected_days,r_per_day,0,0,0,"時間効率不足")

    pwin = _pwin_proxy(feat, sector_rank, entry)
    ev = float(pwin * rr - (1.0 - pwin) * 1.0)

    # EV足切り（仕様：0.4R相当を簡易化して 0.4 で固定）
    if ev < 0.4:
        return EvalPack(stop,tp1,tp2,rr,expected_days,r_per_day,pwin,ev,0,"EV不足")

    adj = float(ev * mkt.regime_multiplier * ev_ctx.event_multiplier)
    return EvalPack(stop,tp1,tp2,rr,expected_days,r_per_day,pwin,ev,adj,"")
