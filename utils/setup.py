from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utils.features import Feat


@dataclass
class SetupResult:
    setup_type: str   # "A" / "B" / "-"
    reason_ng: str    # NG理由（通過時は ""）

    # A/B共通
    trend_ok: bool
    pullback_ok: bool
    breakout_ok: bool
    volume_ok: bool


def judge_setup(feat: Feat) -> SetupResult:
    """仕様書v2.0の Setup A/B 判定（逆張りOFF）。"""
    if feat is None:
        return SetupResult("-", "データ不足", False, False, False, False)

    c = feat.close
    ma20 = feat.ma20
    ma50 = feat.ma50
    slope = feat.ma20_slope_5d
    rsi14 = feat.rsi14
    atr = feat.atr14

    trend_ok = bool(np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50) and c > ma20 > ma50 and slope > 0)
    # 押し目：abs(Close-MA20) <= 0.8ATR & RSI 40-62
    pullback_ok = bool(trend_ok and np.isfinite(atr) and abs(c - ma20) <= 0.8 * atr and 40 <= rsi14 <= 62)

    # ブレイク：Close > HH20 & Volume >= 1.5*VolMA20
    breakout_ok = bool(trend_ok and np.isfinite(feat.hh20) and c > feat.hh20)
    volume_ok = bool(np.isfinite(feat.volume) and np.isfinite(feat.vol_ma20) and feat.vol_ma20 > 0 and feat.volume >= 1.5 * feat.vol_ma20)

    # Setup選択（A優先）
    if pullback_ok:
        return SetupResult("A", "", trend_ok, pullback_ok, breakout_ok, volume_ok)
    if breakout_ok and volume_ok:
        return SetupResult("B", "", trend_ok, pullback_ok, breakout_ok, volume_ok)

    # NG理由（最短で）
    if not trend_ok:
        return SetupResult("-", "トレンド条件NG", trend_ok, pullback_ok, breakout_ok, volume_ok)
    if breakout_ok and not volume_ok:
        return SetupResult("-", "出来高不足", trend_ok, pullback_ok, breakout_ok, volume_ok)
    return SetupResult("-", "形が弱い", trend_ok, pullback_ok, breakout_ok, volume_ok)