from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TradeLevels:
    entry: float
    sl: float
    tp1: float
    tp2: float


def compute_levels(entry: float, atr: float, rr_target: float) -> TradeLevels:
    # v2.3: SL = entry - 1.2ATR
    sl = entry - 1.2 * atr
    r_per_share = max(entry - sl, 1e-9)
    tp1 = entry + 1.5 * r_per_share
    tp2 = entry + rr_target * r_per_share
    return TradeLevels(entry=entry, sl=sl, tp1=tp1, tp2=tp2)


def rr(levels: TradeLevels) -> float:
    r_per_share = max(levels.entry - levels.sl, 1e-9)
    return (levels.tp2 - levels.entry) / r_per_share


def expected_days(entry: float, tp2: float, atr: float) -> float:
    return max(0.5, (tp2 - entry) / max(atr, 1e-9))


def speed_r_per_day(rr_value: float, exp_days: float) -> float:
    return rr_value / max(exp_days, 0.5)


def structural_ev(rr_value: float, trend_strength: float, pullback_quality: float) -> float:
    # v2.3: 因子圧縮（正統）
    return rr_value * trend_strength * pullback_quality


def adj_ev_from_struct(struct_ev_raw: float) -> float:
    # v2.3: スケール正規化（表示・閾値0.50と整合）
    v = struct_ev_raw * 0.35
    return float(np.clip(v, -0.50, 1.20))
