from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.setup import atr14


@dataclass
class TradePlan:
    rr: float
    pwin: float
    ev: float
    adjev: float
    expected_days: float
    r_per_day: float
    entry_low: float
    entry_high: float
    entry_mid: float
    sl: float
    tp1: float
    tp2: float
    r_value: float
    note: str


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def rr_min_by_market(mkt_score: int) -> float:
    """
    仕様：RR下限は地合い連動
      MarketScore ≥ 70 → RR ≥ 1.8
      MarketScore ≥ 60 → RR ≥ 2.0
      MarketScore ≥ 50 → RR ≥ 2.2
      MarketScore < 50 → RR ≥ 2.5
    """
    if mkt_score >= 70:
        return 1.8
    if mkt_score >= 60:
        return 2.0
    if mkt_score >= 50:
        return 2.2
    return 2.5


def _pwin_base(setup: str) -> float:
    # ログ無し前提の仮定（仕様：将来ログで更新）
    if setup == "A1":
        return 0.48
    if setup == "A2":
        return 0.44
    if setup == "B":
        return 0.40
    return 0.33


def build_trade_plan(
    hist: pd.DataFrame,
    setup: str,
    entry_low: float,
    entry_high: float,
    entry_mid: float,
    stop_seed: float,
    mkt_score: int,
    macro_on: bool,
) -> Optional[TradePlan]:
    """
    仕様のExit:
      STOP: IN - 1.2ATR（seed） / 直近安値が近い場合はそちら優先
      TP1 : +1.5R
      TP2 : +2.0〜3.5R（可変）

    速度:
      ExpectedDays = (TP2 - IN) / ATR
      R/day = RR / ExpectedDays

    EV:
      EV = Pwin*RR - (1-Pwin)
      AdjEV は Macro警戒で減衰
    """
    if hist is None or hist.empty or len(hist) < 80:
        return None

    df = hist.copy()
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    atr = atr14(df)
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(entry_mid) and entry_mid > 0):
        return None

    # SL finalize
    sl = float(stop_seed)
    lookback = 12
    swing_low = _safe_float(low.tail(lookback).min(), np.nan)
    if np.isfinite(swing_low):
        # 直近安値が近い（=高い）なら採用（損切り幅を広げない）
        cand = float(swing_low - 0.2 * atr)
        if cand < entry_mid and cand > sl:
            sl = cand

    r_value = float(entry_mid - sl)
    if r_value <= 0:
        return None

    # SLが近すぎるとノイズで刈られるので最低幅を確保
    if r_value < 0.6 * atr:
        sl = float(entry_mid - 0.6 * atr)
        r_value = float(entry_mid - sl)

    # TP2 multiple (2.0..3.5)
    # A1:伸びやすい想定 / B:控えめ / Macro時は控えめ
    rr_target = 2.6
    if setup == "A1":
        rr_target = 3.0 if mkt_score >= 60 else 2.7
    elif setup == "A2":
        rr_target = 2.6 if mkt_score >= 60 else 2.4
    elif setup == "B":
        rr_target = 2.4 if mkt_score >= 60 else 2.2

    if macro_on:
        rr_target = max(2.0, rr_target - 0.4)

    rr_target = float(np.clip(rr_target, 2.0, 3.5))

    tp1 = float(entry_mid + 1.5 * r_value)
    tp2 = float(entry_mid + rr_target * r_value)

    # TP2 cap by 60d high (realistic)
    hi_window = 60 if len(high) >= 60 else len(high)
    high_60 = _safe_float(high.tail(hi_window).max(), np.nan)
    if np.isfinite(high_60) and high_60 > 0:
        tp2 = min(tp2, float(high_60 * 0.995))
        if tp2 <= tp1:
            tp2 = float(tp1 * 1.05)

    rr = float((tp2 - entry_mid) / r_value)

    expected_days = float((tp2 - entry_mid) / atr)
    if not np.isfinite(expected_days) or expected_days <= 0:
        return None

    rday = float(rr / expected_days)

    pwin = _pwin_base(setup)
    pwin += float((mkt_score - 50) * 0.001)
    if macro_on:
        pwin -= 0.05
    pwin = float(np.clip(pwin, 0.25, 0.60))

    ev = float(pwin * rr - (1.0 - pwin))
    adjev = float(ev * (0.85 if macro_on else 1.0))

    return TradePlan(
        rr=float(rr),
        pwin=float(pwin),
        ev=float(ev),
        adjev=float(adjev),
        expected_days=float(expected_days),
        r_per_day=float(rday),
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        entry_mid=float(entry_mid),
        sl=float(sl),
        tp1=float(tp1),
        tp2=float(tp2),
        r_value=float(r_value),
        note="",
    )
