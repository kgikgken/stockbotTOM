# utils/setup.py
from __future__ import annotations

from dataclasses import dataclass

from utils.features import Tech


@dataclass(frozen=True)
class SetupResult:
    ok: bool
    kind: str  # "A1" / "A2" / "-" など
    reason: str


def decide_setup(tech: Tech) -> SetupResult:
    # 大前提：順張りのみ（MA構造）
    if not (tech.close > tech.sma20 > tech.sma50):
        return SetupResult(False, "-", "MA構造が上向きではない")

    # RSI過熱なし（基準）
    if not (40 <= tech.rsi <= 62):
        return SetupResult(False, "-", "RSI条件外")

    # 押し目位置（SMA20〜SMA50付近）
    dist20 = abs(tech.close - tech.sma20)
    dist50 = abs(tech.close - tech.sma50)
    if tech.atr <= 0:
        return SetupResult(False, "-", "ATR不正")

    # A1: SMA20寄りの浅い押し目
    if dist20 <= 0.8 * tech.atr and (tech.close >= tech.sma20):
        return SetupResult(True, "A1", "")

    # A2: 深め押し目（SMA50寄りまで押すが構造維持）
    if dist50 <= 1.0 * tech.atr and (tech.close >= tech.sma50):
        return SetupResult(True, "A2", "")

    return SetupResult(False, "-", "押し目位置が条件外")