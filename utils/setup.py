# ============================================
# utils/setup.py
# Setup判定（AをA1/A2に分離）
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from utils.features import Features


@dataclass
class SetupResult:
    setup_type: str  # "A1" "A2" "B" or "-"
    reason: str


def is_trend_up(f: Features) -> bool:
    return (f.close > f.sma20 > f.sma50) and (f.sma20 - f.sma50) > 0.0


def setup_a1(f: Features) -> Optional[SetupResult]:
    """
    A1: 強トレンド + 浅い押し目（MA20近辺）+ RSI過熱なし
    """
    if not is_trend_up(f):
        return None
    # 押し目: SMA20から0.8ATR以内
    if f.atr14 <= 0:
        return None
    if abs(f.close - f.sma20) > 0.8 * f.atr14:
        return None
    if not (40.0 <= f.rsi14 <= 62.0):
        return None
    return SetupResult(setup_type="A1", reason="トレンド上 + MA20付近の押し目")


def setup_a2(f: Features) -> Optional[SetupResult]:
    """
    A2: トレンドは上だが、押しが深め（MA50寄り） or RSIがやや低め
    """
    if not (f.close > f.sma50):
        return None
    if f.atr14 <= 0:
        return None

    # MA20割れは許容、ただしMA50は維持
    if f.close < f.sma50:
        return None

    # MA20〜MA50の間で押し目判定（深め）
    # 近さ: min(|close-sma20|, |close-sma50|) <= 1.2ATR
    near = min(abs(f.close - f.sma20), abs(f.close - f.sma50))
    if near > 1.2 * f.atr14:
        return None

    # RSIは過熱なし（35〜60）
    if not (35.0 <= f.rsi14 <= 60.0):
        return None

    # A1に該当するならA1優先
    if setup_a1(f) is not None:
        return None

    return SetupResult(setup_type="A2", reason="トレンド上 + 深めの押し目（MA50寄り）")


def detect_setup(f: Features) -> SetupResult:
    a1 = setup_a1(f)
    if a1:
        return a1
    a2 = setup_a2(f)
    if a2:
        return a2
    return SetupResult(setup_type="-", reason="形不一致")